from pathlib import Path

from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config_panda.yaml"


if __name__ == '__main__':
    panda = PandaGraspInterface(str(DEFAULT_CONFIG_PATH))

    # take a fist guess at where the world frame might be
    panda.robot.move_to_start()

    joints = [panda.get_jj()]

    while True:
        ii = input("Press Enter to record joints, anything else to exix: ")
        if len(ii) > 0: break
        panda.print_jj()
        joints.append(panda.get_jj())

    for joint in joints:
        formatted = ", ".join(f"{value:.3f}" for value in joint)
        print(f"[{formatted}]")
