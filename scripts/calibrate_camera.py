"""Calibrate the wrist camera and workspace frame."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from omegaconf import OmegaConf

from calibration_common import collect_samples, hand_eye, mean_transform
from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.utils.transform import Rotation, Transform


ROOT = Path(__file__).resolve().parents[1]
PANDA_CONFIG = ROOT / "configs/config_panda.yaml"
WORKSPACE_CONFIG = ROOT / "configs/config_workspace.yaml"
PANDA_HOSTNAME = "172.16.0.2"
CAMERA_TYPE = "camera_fdm"
TAG_ORIGIN_IN_WORLD = np.array([0.15, 0.15, 0.0])


def create_panda() -> PandaGraspInterface:
    config = OmegaConf.create({
        "hostname": PANDA_HOSTNAME,
        "camera_type": CAMERA_TYPE,
    })
    with NamedTemporaryFile(mode="w", suffix=".yaml") as temporary:
        OmegaConf.save(config, temporary.name)
        return PandaGraspInterface(temporary.name)


def main() -> None:
    panda = create_panda()
    try:
        ee_to_base, target_to_camera = collect_samples(
            panda, WORKSPACE_CONFIG, "workspace"
        )
        camera_to_ee = hand_eye(ee_to_base, target_to_camera)
        target_to_base = mean_transform([
            ee * camera_to_ee * target
            for ee, target in zip(ee_to_base, target_to_camera)
        ])
        target_to_world = Transform(Rotation.identity(), TAG_ORIGIN_IN_WORLD)
        world_to_base = target_to_base * target_to_world.inverse()
    finally:
        panda.camera.finalize()
        cv2.destroyAllWindows()

    config = OmegaConf.load(PANDA_CONFIG)
    config.hostname = PANDA_HOSTNAME
    config.camera_type = CAMERA_TYPE
    config.T_camera2ee = camera_to_ee.to_list()
    config.T_world2panda = world_to_base.to_list()
    OmegaConf.save(config, PANDA_CONFIG)
    print(f"Calibration complete. Wrote {PANDA_CONFIG}")


if __name__ == "__main__":
    main()
