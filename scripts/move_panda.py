
import os
from pathlib import Path

import cv2
import numpy as np

from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.panda_constants import SCANNING_JOINTS_DEFAULT


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config_panda.yaml"


def main():
    panda = PandaGraspInterface(str(DEFAULT_CONFIG_PATH))

    panda.move_to_start()
    panda.move_jj(SCANNING_JOINTS_DEFAULT[0])

if __name__ == "__main__":
    main()
