"""Calibrate either a wrist-mounted or fixed camera with AprilTag bundles."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

import cv2
import numpy as np
from omegaconf import OmegaConf

from simple_panda_grasp.panda_constants import WRIST_CALIBRATION_JOINTS
from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.utils.transform import Rotation, Transform


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CONFIG = ROOT / "configs" / "config_panda.yaml"
WORKSPACE_CONFIG = ROOT / "configs" / "config_workspace.yaml"
CALIBRATION_BUNDLE_CONFIG = ROOT / "configs" / "config_calibration_bundle.yaml"

PANDA_HOSTNAME = "172.16.0.2"
CAMERA_TYPE = "camera_fdm"
TAG_ORIGIN_IN_WORLD = np.array([0.15, 0.15, 0.0])
MIN_SAMPLES = 6


def load_target(path: Path, bundle_name: str):
    config = OmegaConf.load(path)
    if config.tag_family != "tag36h11":
        raise ValueError(f"Unsupported AprilTag family: {config.tag_family}")

    bundles = [bundle for bundle in config.tag_bundles if bundle.name == bundle_name]
    if len(bundles) != 1:
        raise ValueError(f"{path.name} must contain one {bundle_name!r} bundle")

    marker_points = {}
    for tag in bundles[0].layout:
        half = tag.size / 2
        corners = np.array([
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ])
        marker_to_target = Transform(
            Rotation.from_quat([tag.qx, tag.qy, tag.qz, tag.qw]),
            [tag.x, tag.y, tag.z],
        )
        marker_points[int(tag.id)] = np.asarray(
            [marker_to_target.transform_point(corner) for corner in corners],
            dtype=np.float32,
        )

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    return dictionary, marker_points


def detect_target(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dictionary: cv2.aruco.Dictionary,
    marker_points: dict[int, np.ndarray],
) -> Transform | None:
    """Return the bundle-to-camera transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    object_points = []
    image_points = []
    for marker_image_corners, marker_id in zip(corners, ids.ravel()):
        marker_id = int(marker_id)
        if marker_id not in marker_points:
            continue
        object_points.append(marker_points[marker_id])
        image_points.append(marker_image_corners.reshape(4, 2))

    if not object_points:
        return None

    success, rotation_vector, translation = cv2.solvePnP(
        np.concatenate(object_points),
        np.concatenate(image_points),
        camera_matrix,
        None,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return Transform(
        Rotation.from_matrix(rotation_matrix),
        translation.reshape(3),
    )


def mean_transform(transforms: list[Transform]) -> Transform:
    rotations = Rotation.from_matrix(
        np.stack([transform.rotation.as_matrix() for transform in transforms])
    ).mean()
    translations = np.mean(
        [transform.translation for transform in transforms], axis=0
    )
    return Transform(rotations, translations)


def capture_target(panda, dictionary, marker_points) -> Transform | None:
    color, _ = panda.camera.capture()
    return detect_target(
        color, panda.camera.camera_matrix, dictionary, marker_points
    )


def collect_samples(panda: PandaGraspInterface, mode: str):
    target_config, bundle_name = {
        "eye-in-hand": (WORKSPACE_CONFIG, "workspace"),
        "eye-on-base": (CALIBRATION_BUNDLE_CONFIG, "calibration"),
    }[mode]
    dictionary, marker_points = load_target(target_config, bundle_name)
    ee_to_base = []
    target_to_camera = []

    if mode == "eye-on-base":
        input("Rigidly attach the calibration prism, then press Enter: ")
        print("Use varied poses and keep two adjacent tags visible when possible.")
    panda.move_to_start()
    poses = WRIST_CALIBRATION_JOINTS

    for index, joints in enumerate(poses, start=1):
        print(f"[{index}/{len(poses)}] Capturing calibration pose")
        if not panda.move_jj(joints):
            print("  Robot could not reach this pose; skipping it.")
            continue
        sleep(3.0)

        target_pose = capture_target(panda, dictionary, marker_points)
        if target_pose is None:
            print("  Calibration target not visible; skipping this pose.")
            continue

        ee_to_base.append(panda.get_ee2panda())
        target_to_camera.append(target_pose)

    if len(ee_to_base) < MIN_SAMPLES:
        raise RuntimeError(
            f"Only {len(ee_to_base)} usable poses were captured; "
            f"at least {MIN_SAMPLES} are required. Check the target placement."
        )
    return ee_to_base, target_to_camera


def hand_eye(robot_poses: list[Transform], target_poses: list[Transform]) -> Transform:
    rotation, translation = cv2.calibrateHandEye(
        [pose.rotation.as_matrix() for pose in robot_poses],
        [pose.translation for pose in robot_poses],
        [pose.rotation.as_matrix() for pose in target_poses],
        [pose.translation for pose in target_poses],
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    )
    result = Transform(
        Rotation.from_matrix(rotation), np.asarray(translation).reshape(3)
    )
    if not np.isfinite(result.to_list()).all():
        raise RuntimeError("OpenCV could not find a valid hand-eye calibration.")
    return result


def calibrate_eye_in_hand(panda: PandaGraspInterface):
    ee_to_base, target_to_camera = collect_samples(panda, "eye-in-hand")
    camera_to_ee = hand_eye(ee_to_base, target_to_camera)

    target_to_panda = mean_transform([
        ee_to_panda * camera_to_ee * target_to_cam
        for ee_to_panda, target_to_cam in zip(ee_to_base, target_to_camera)
    ])
    target_to_world = Transform(Rotation.identity(), TAG_ORIGIN_IN_WORLD)
    world_to_panda = target_to_panda * target_to_world.inverse()
    return camera_to_ee, world_to_panda


def calibrate_eye_on_base(panda: PandaGraspInterface) -> Transform:
    ee_to_base, target_to_camera = collect_samples(panda, "eye-on-base")
    # Inverting the robot poses maps OpenCV's hand-eye result to camera -> base.
    base_to_ee = [pose.inverse() for pose in ee_to_base]
    return hand_eye(base_to_ee, target_to_camera)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", nargs="?", choices=("eye-in-hand", "eye-on-base"),
        default="eye-in-hand",
    )
    mode = parser.parse_args().mode
    bootstrap_config = OmegaConf.create({
        "hostname": PANDA_HOSTNAME,
        "camera_type": CAMERA_TYPE,
    })
    with NamedTemporaryFile(mode="w", suffix=".yaml") as temporary_config:
        OmegaConf.save(bootstrap_config, temporary_config.name)
        panda = PandaGraspInterface(temporary_config.name)

    try:
        if mode == "eye-in-hand":
            camera_to_ee, world_to_panda = calibrate_eye_in_hand(panda)
        else:
            camera_to_panda = calibrate_eye_on_base(panda)
    finally:
        panda.camera.finalize()

    config = (
        OmegaConf.load(OUTPUT_CONFIG)
        if OUTPUT_CONFIG.exists()
        else OmegaConf.create()
    )
    config.hostname = PANDA_HOSTNAME
    config.camera_type = CAMERA_TYPE
    if mode == "eye-in-hand":
        config.T_camera2ee = camera_to_ee.to_list()
        config.T_world2panda = world_to_panda.to_list()
    else:
        config.T_camera2panda = camera_to_panda.to_list()
    OmegaConf.save(config, OUTPUT_CONFIG)
    print(f"{mode} calibration complete. Wrote {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
