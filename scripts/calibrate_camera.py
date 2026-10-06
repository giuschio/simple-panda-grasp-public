# Print configs/workspace.pdf at actual size on A3 paper, place it flat on the table in front of the Panda, then run: python scripts/calibrate_camera.py
"""Calibrate the wrist camera and workspace frame in one pass."""

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

PANDA_HOSTNAME = "172.16.0.2"
CAMERA_TYPE = "camera_fdm"
TAG_ORIGIN_IN_WORLD = np.array([0.15, 0.15, 0.0])
MIN_SAMPLES = 6


def load_target() -> tuple[cv2.aruco.Dictionary, dict[int, np.ndarray]]:
    config = OmegaConf.load(WORKSPACE_CONFIG)
    if config.tag_family != "tag36h11":
        raise ValueError(f"Unsupported AprilTag family: {config.tag_family}")

    bundles = [bundle for bundle in config.tag_bundles if bundle.name == "workspace"]
    if len(bundles) != 1:
        raise ValueError("config_workspace.yaml must contain one 'workspace' bundle")

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
    """Return the workspace-target-to-camera transform."""
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


def calibrate(panda: PandaGraspInterface) -> tuple[Transform, Transform]:
    gripper_to_base = []
    target_to_camera = []
    dictionary, marker_points = load_target()

    panda.move_to_start()
    for index, joints in enumerate(WRIST_CALIBRATION_JOINTS, start=1):
        print(f"[{index}/{len(WRIST_CALIBRATION_JOINTS)}] Capturing calibration pose")
        if not panda.move_jj(joints):
            print("  Robot could not reach this pose; skipping it.")
            continue

        sleep(3.0)
        color, _ = panda.camera.capture()
        target_pose = detect_target(
            color,
            panda.camera.camera_matrix,
            dictionary,
            marker_points,
        )
        if target_pose is None:
            print("  Workspace target not visible; skipping this pose.")
            continue

        gripper_to_base.append(panda.get_ee2panda())
        target_to_camera.append(target_pose)

    if len(gripper_to_base) < MIN_SAMPLES:
        raise RuntimeError(
            f"Only {len(gripper_to_base)} usable poses were captured; "
            f"at least {MIN_SAMPLES} are required. Check the target placement."
        )

    camera_rotation, camera_translation = cv2.calibrateHandEye(
        [pose.rotation.as_matrix() for pose in gripper_to_base],
        [pose.translation for pose in gripper_to_base],
        [pose.rotation.as_matrix() for pose in target_to_camera],
        [pose.translation for pose in target_to_camera],
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    )
    camera_to_ee = Transform(
        Rotation.from_matrix(camera_rotation),
        np.asarray(camera_translation).reshape(3),
    )
    if not np.isfinite(camera_to_ee.to_list()).all():
        raise RuntimeError("OpenCV could not find a valid hand-eye calibration.")

    target_to_panda = mean_transform([
        ee_to_panda * camera_to_ee * target_to_cam
        for ee_to_panda, target_to_cam in zip(gripper_to_base, target_to_camera)
    ])
    target_to_world = Transform(Rotation.identity(), TAG_ORIGIN_IN_WORLD)
    world_to_panda = target_to_panda * target_to_world.inverse()
    return camera_to_ee, world_to_panda


def main() -> None:
    bootstrap_config = OmegaConf.create({
        "hostname": PANDA_HOSTNAME,
        "camera_type": CAMERA_TYPE,
    })
    with NamedTemporaryFile(mode="w", suffix=".yaml") as temporary_config:
        OmegaConf.save(bootstrap_config, temporary_config.name)
        panda = PandaGraspInterface(temporary_config.name)

    try:
        camera_to_ee, world_to_panda = calibrate(panda)
    finally:
        panda.camera.finalize()

    config = OmegaConf.create({
        "hostname": PANDA_HOSTNAME,
        "camera_type": CAMERA_TYPE,
        "T_camera2ee": camera_to_ee.to_list(),
        "T_world2panda": world_to_panda.to_list(),
    })
    OmegaConf.save(config, OUTPUT_CONFIG)
    print(f"Calibration complete. Wrote {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
