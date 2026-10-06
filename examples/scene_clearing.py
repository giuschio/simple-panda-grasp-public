"""Repeatedly sample and execute grasps on a real Panda."""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

from scene_clearing_utils import EdgeGraspSampler, visualize_grasps
from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.utils.perception_utils import cloud_from_image_data
from simple_panda_grasp.utils.offline_visualization import visualize_scene_pcd

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config_panda.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--release-position", choices=("front", "side"), default="front")
    parser.add_argument("--show-grasp-proposals", default=True)
    parser.add_argument("--save-scene-pcd-location", default="saves", help="If None, does not save anything. Otherwise, interpreted as a folder name")
    args = parser.parse_args()

    sampler = EdgeGraspSampler()
    panda_interface = PandaGraspInterface(args.config_path)
    if args.save_scene_pcd_location is not None:
        Path(args.save_scene_pcd_location).mkdir(parents=True, exist_ok=True)
    scene_idx = 0
    while True:
        image_data = panda_interface.get_observation_default_viewpoint()
        pointcloud = cloud_from_image_data(
            image_data,
            lower_bounds=panda_interface.lower,
            upper_bounds=panda_interface.upper,
        )
        if args.save_scene_pcd_location is not None:
            scene_dir = Path(args.save_scene_pcd_location) / f"scene_{scene_idx}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            i = image_data["intrinsics"]
            data = {
                "rgb": image_data["rgb_imgs"],
                "depth": image_data["depth_imgs"],
                "intrinsics": np.array([i.width, i.height, i.fx, i.fy, i.cx, i.cy]),
                "extrinsics": image_data["extrinsics"],
                "eye": image_data["eye"],
            }
            o3d.io.write_point_cloud(scene_dir / "pointcloud.ply", pointcloud)
            for name, value in data.items():
                np.save(scene_dir / f"{name}.npy", value)
            for view_idx, rgb in enumerate(data["rgb"]):
                o3d.io.write_image(str(scene_dir / f"rgb_{view_idx}.png"), o3d.geometry.Image(rgb))
            np.savez_compressed(scene_dir / "image_data.npz", **data)
        grasp_proposals = panda_interface.run_grasp_checks(
            sampler(pointcloud, n_grasps=100),
            refine_pose=True,
        )

        if args.show_grasp_proposals:
            visualize_scene_pcd(pointcloud, grasp_proposals)

        success, message = panda_interface.pick(grasp_proposals[0])
        print(f"success={success}: {message}")
        panda_interface.place(args.release_position)
        scene_idx += 1

        if input("Press Enter for another grasp, or q to quit: ").strip().lower() == "q":
            break
    panda_interface.move_to_start()
    panda_interface.camera.finalize()


if __name__ == "__main__":
    main()
