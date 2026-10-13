"""Calibrate either a wrist-mounted or fixed camera with AprilTag bundles."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

import cv2
import numpy as np
from omegaconf import OmegaConf

from apriltag_bundle_detector.solver import AprilTagBundleDetector
from simple_panda_grasp.panda_constants import WRIST_CALIBRATION_JOINTS
from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.utils.transform import Rotation, Transform


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CONFIG = ROOT / "configs" / "config_panda.yaml"
WORKSPACE_CONFIG = ROOT / "configs" / "config_workspace.yaml"
CALIBRATION_BUNDLE_CONFIG = ROOT / "configs" / "config_calibration_bundle.yaml"

PANDA_HOSTNAME = "172.16.0.2"
CAMERA_TYPE = "camera_raw"
TAG_ORIGIN_IN_WORLD = np.array([0.15, 0.15, 0.0])
MIN_SAMPLES = 6
MAX_CAMERA_TO_BASE_DISTANCE = 3.0
MAX_TRANSLATION_RMS = 0.03
MAX_ROTATION_RMS_DEG = 3.0
PREVIEW_WINDOW = "Calibration bundle detection"


def draw_bundle_frame(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    bundle_to_camera: Transform,
    axis_length: float,
    bundle_name: str,
) -> np.ndarray:
    """Draw the detected bundle coordinate frame on a camera image."""
    axes = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float32,
    )
    image_points, _ = cv2.projectPoints(
        axes,
        bundle_to_camera.rotation.as_rotvec(),
        bundle_to_camera.translation,
        camera_matrix,
        None,
    )
    points = np.rint(image_points.reshape(-1, 2)).astype(int)
    origin = tuple(points[0])
    colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    labels = ("X", "Y", "Z")
    for endpoint, color, label in zip(points[1:], colors, labels):
        endpoint_tuple = tuple(endpoint)
        cv2.arrowedLine(
            image, origin, endpoint_tuple, color, 3, cv2.LINE_AA, tipLength=0.15
        )
        cv2.putText(
            image,
            label,
            endpoint_tuple,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.circle(image, origin, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        bundle_name,
        (origin[0] + 8, origin[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def mean_transform(transforms: list[Transform]) -> Transform:
    rotations = Rotation.from_matrix(
        np.stack([transform.rotation.as_matrix() for transform in transforms])
    ).mean()
    translations = np.mean(
        [transform.translation for transform in transforms], axis=0
    )
    return Transform(rotations, translations)


def capture_target(
    panda: PandaGraspInterface,
    detector: AprilTagBundleDetector,
    bundle_name: str,
) -> Transform | None:
    color, _ = panda.camera.capture()
    target_pose = detector(color, debug=True).get(bundle_name)
    preview = color.copy()
    if target_pose is None:
        cv2.putText(
            preview,
            f"{bundle_name}: not detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        tag_sizes = [
            tag["size"] for tag in detector.bundles[bundle_name].values()
        ]
        axis_length = 1.5 * max(tag_sizes, default=0.05)
        draw_bundle_frame(
            preview,
            panda.camera.camera_matrix,
            target_pose,
            axis_length,
            bundle_name,
        )
    cv2.imshow(PREVIEW_WINDOW, preview)
    cv2.waitKey(1)
    return target_pose


def collect_samples(panda: PandaGraspInterface, mode: str):
    target_config, bundle_name = {
        "eye-in-hand": (WORKSPACE_CONFIG, "workspace"),
        "eye-on-base": (CALIBRATION_BUNDLE_CONFIG, "calibration"),
    }[mode]
    detector = AprilTagBundleDetector(
        str(target_config),
        panda.camera.camera_matrix,
        reproj_error_thresh=1.0,
        min_tags=1,
    )
    ee_to_base = []
    target_to_camera = []

    if mode == "eye-on-base":
        input("Rigidly attach the calibration prism, then press Enter: ")
        print("Use varied poses and keep two adjacent tags visible when possible.")
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    panda.move_to_start()
    poses = WRIST_CALIBRATION_JOINTS

    for index, joints in enumerate(poses, start=1):
        print(f"[{index}/{len(poses)}] Capturing calibration pose")
        if not panda.move_jj(joints):
            print("  Robot could not reach this pose; skipping it.")
            continue
        sleep(3.0)

        target_pose = capture_target(panda, detector, bundle_name)
        if target_pose is None:
            print("  Calibration target not visible; skipping this pose.")
            continue

        target_distance = np.linalg.norm(target_pose.translation)
        print(f"  Target distance from camera: {target_distance:.3f} m")
        if not 0.1 < target_distance < 3.0:
            print("  Implausible target pose; skipping this sample.")
            continue

        ee_to_base.append(panda.get_ee2panda())
        target_to_camera.append(target_pose)

    if len(ee_to_base) < MIN_SAMPLES:
        raise RuntimeError(
            f"Only {len(ee_to_base)} usable poses were captured; "
            f"at least {MIN_SAMPLES} are required. Check the target placement."
        )
    return ee_to_base, target_to_camera


def calibration_residual(
    robot_poses: list[Transform],
    target_poses: list[Transform],
    camera_to_robot: Transform,
) -> tuple[float, float]:
    """Measure how constant the rigidly mounted calibration target remains."""
    fixed_target_poses = [
        robot_pose * camera_to_robot * target_pose
        for robot_pose, target_pose in zip(robot_poses, target_poses)
    ]
    average = mean_transform(fixed_target_poses)
    translation_errors = [
        np.linalg.norm(pose.translation - average.translation)
        for pose in fixed_target_poses
    ]
    rotation_errors = [
        (average.rotation.inv() * pose.rotation).magnitude()
        for pose in fixed_target_poses
    ]
    translation_rms = float(np.sqrt(np.mean(np.square(translation_errors))))
    rotation_rms_deg = float(
        np.rad2deg(np.sqrt(np.mean(np.square(rotation_errors))))
    )
    return translation_rms, rotation_rms_deg


def hand_eye(robot_poses: list[Transform], target_poses: list[Transform]) -> Transform:
    methods = {
        "Tsai": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    candidates = []
    for name, method in methods.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                [pose.rotation.as_matrix() for pose in robot_poses],
                [pose.translation for pose in robot_poses],
                [pose.rotation.as_matrix() for pose in target_poses],
                [pose.translation for pose in target_poses],
                method=method,
            )
            result = Transform(
                Rotation.from_matrix(rotation),
                np.asarray(translation).reshape(3),
            )
            if not np.isfinite(result.to_list()).all():
                continue
            translation_rms, rotation_rms_deg = calibration_residual(
                robot_poses, target_poses, result
            )
        except (ValueError, cv2.error):
            continue

        # Treat one radian of orientation inconsistency like 10 cm of position
        # inconsistency. This selects a stable solution without hiding either
        # residual from the operator.
        score = translation_rms + 0.1 * np.deg2rad(rotation_rms_deg)
        candidates.append(
            (score, name, result, translation_rms, rotation_rms_deg)
        )
        print(
            f"  {name:10s}: translation RMS {translation_rms * 1000:6.1f} mm, "
            f"rotation RMS {rotation_rms_deg:5.2f} deg, "
            f"translation norm {np.linalg.norm(result.translation):.3f} m"
        )

    if not candidates:
        raise RuntimeError("OpenCV could not find a finite hand-eye calibration.")

    _, name, result, translation_rms, rotation_rms_deg = min(
        candidates, key=lambda candidate: candidate[0]
    )
    print(f"Selected {name} hand-eye solution.")
    if (
        translation_rms > MAX_TRANSLATION_RMS
        or rotation_rms_deg > MAX_ROTATION_RMS_DEG
    ):
        raise RuntimeError(
            "Calibration is inconsistent and was not saved: "
            f"{translation_rms * 1000:.1f} mm translation RMS, "
            f"{rotation_rms_deg:.2f} deg rotation RMS. Check that the prism "
            "is rigid, its tag IDs match config_calibration_bundle.yaml, and "
            "that the captured poses include substantial rotations."
        )
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
    camera_to_base = hand_eye(base_to_ee, target_to_camera)
    distance = np.linalg.norm(camera_to_base.translation)
    if distance > MAX_CAMERA_TO_BASE_DISTANCE:
        raise RuntimeError(
            "Calibration was not saved because the estimated camera is "
            f"{distance:.2f} m from the Panda base. This is almost certainly "
            "a failed solve."
        )
    return camera_to_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", nargs="?", choices=("eye-in-hand", "eye-on-base"),
        default="eye-on-base",
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
        cv2.destroyAllWindows()

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
