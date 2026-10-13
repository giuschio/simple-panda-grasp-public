"""Visualize a configured fixed-camera-to-Panda-base transform."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.utils.transform import Transform


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/config_panda.yaml"
VOXEL_SIZE = 0.004
MAX_DEPTH = 1.5
EE_FRAME_SIZE = 0.07
BASE_FRAME_SIZE = 0.14


def capture_cloud(camera) -> o3d.geometry.PointCloud:
    """Capture a colored point cloud expressed in the camera frame."""
    color, depth = camera.capture()
    depth = np.asarray(depth, dtype=float)
    color = np.asarray(color)[..., ::-1]
    intrinsics = camera.color_intrinsics

    ys, xs = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0) & (depth < MAX_DEPTH)
    if not np.any(valid):
        raise RuntimeError("The camera returned no valid depth samples.")

    points = np.stack(
        (
            (xs - intrinsics.cx) * depth / intrinsics.fx,
            (ys - intrinsics.cy) * depth / intrinsics.fy,
            depth,
        ),
        axis=-1,
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[valid])
    cloud.colors = o3d.utility.Vector3dVector(color[valid] / 255.0)
    return cloud.voxel_down_sample(VOXEL_SIZE)


def frame(transform: Transform, size: float) -> o3d.geometry.TriangleMesh:
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    axes.transform(transform.as_matrix())
    return axes


def load_camera_to_base(config_path: Path) -> Transform:
    config = OmegaConf.load(config_path)
    if "T_camera2panda" not in config:
        raise RuntimeError(
            f"{config_path} does not contain T_camera2panda. "
            "Run the fixed-camera calibration first."
        )

    values = np.asarray(config.T_camera2panda, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError(
            "T_camera2panda must contain seven finite values in "
            "[qx, qy, qz, qw, tx, ty, tz] order."
        )
    return Transform.from_list(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the fixed-camera cloud and display the configured Panda "
            "base frame and current end-effector frame in camera coordinates."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Panda configuration file (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_to_base = load_camera_to_base(args.config)
    base_to_camera = camera_to_base.inverse()

    panda = PandaGraspInterface(str(args.config))
    try:
        input("Position the robot for the test, then press Enter to capture: ")
        scene = capture_cloud(panda.camera)
        ee_to_base = panda.get_ee2panda()
        ee_to_camera = base_to_camera * ee_to_base

        print(
            "Displaying the current EE frame (small) and Panda base frame "
            "(large). Axes are X=red, Y=green, Z=blue."
        )
        o3d.visualization.draw_geometries(
            [
                scene,
                frame(ee_to_camera, EE_FRAME_SIZE),
                frame(base_to_camera, BASE_FRAME_SIZE),
            ],
            window_name="Camera-to-base test (EE small, base large)",
        )
    finally:
        panda.camera.finalize()


if __name__ == "__main__":
    main()
