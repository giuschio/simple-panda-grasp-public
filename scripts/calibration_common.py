"""Shared AprilTag hand-eye calibration helpers."""

from pathlib import Path
from time import sleep

import cv2
import numpy as np

from apriltag_bundle_detector.solver import AprilTagBundleDetector
from simple_panda_grasp.panda_constants import WRIST_CALIBRATION_JOINTS
from simple_panda_grasp.utils.transform import Rotation, Transform


MIN_SAMPLES = 6
PREVIEW_WINDOW = "Calibration bundle detection"


def mean_transform(transforms: list[Transform]) -> Transform:
    rotations = Rotation.from_matrix(
        np.stack([transform.rotation.as_matrix() for transform in transforms])
    ).mean()
    translations = np.mean(
        [transform.translation for transform in transforms], axis=0
    )
    return Transform(rotations, translations)


def draw_bundle_frame(image, camera, transform, axis_length, name) -> None:
    axes = np.array([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length],
    ], dtype=np.float32)
    image_points, _ = cv2.projectPoints(
        axes,
        transform.rotation.as_rotvec(),
        transform.translation,
        camera.camera_matrix,
        getattr(camera, "dist_coeffs", None),
    )
    points = np.rint(image_points.reshape(-1, 2)).astype(int)
    origin = tuple(points[0])
    for endpoint, color in zip(
        points[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    ):
        cv2.arrowedLine(
            image, origin, tuple(endpoint), color, 3, cv2.LINE_AA, tipLength=0.15
        )
    cv2.putText(
        image, name, (origin[0] + 8, origin[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


def capture_bundle(panda, detector, bundle_name: str) -> Transform | None:
    color, _ = panda.camera.capture()
    detected = detector(color, debug=True).get(bundle_name)
    preview = color.copy()
    if detected is None:
        cv2.putText(
            preview, f"{bundle_name}: not detected", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
        )
        result = None
    else:
        result = Transform.from_matrix(detected.as_matrix())
        sizes = [tag["size"] for tag in detector.bundles[bundle_name].values()]
        draw_bundle_frame(
            preview, panda.camera, result,
            1.5 * max(sizes, default=0.05), bundle_name,
        )
    cv2.imshow(PREVIEW_WINDOW, preview)
    cv2.waitKey(1)
    return result


def collect_samples(panda, config_path: Path, bundle_name: str):
    detector = AprilTagBundleDetector(
        str(config_path),
        panda.camera.camera_matrix,
        distortion_coeffs=getattr(panda.camera, "dist_coeffs", None),
        reproj_error_thresh=1.0,
        min_tags=1,
    )
    if bundle_name not in detector.bundles:
        raise ValueError(f"Bundle {bundle_name!r} is not defined in {config_path}")

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    panda.move_to_start()
    robot_poses, target_poses = [], []
    for index, joints in enumerate(WRIST_CALIBRATION_JOINTS, start=1):
        print(f"[{index}/{len(WRIST_CALIBRATION_JOINTS)}] Capturing pose")
        if not panda.move_jj(joints):
            print("  Robot could not reach this pose; skipping it.")
            continue
        sleep(3.0)
        target_pose = capture_bundle(panda, detector, bundle_name)
        if target_pose is None:
            print("  Bundle not visible; skipping this pose.")
            continue
        print(f"  Bundle distance: {np.linalg.norm(target_pose.translation):.3f} m")
        robot_poses.append(panda.get_ee2panda())
        target_poses.append(target_pose)

    if len(robot_poses) < MIN_SAMPLES:
        raise RuntimeError(
            f"Only {len(robot_poses)} usable poses; at least {MIN_SAMPLES} are required."
        )
    return robot_poses, target_poses


def hand_eye(robot_poses, target_poses) -> Transform:
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
