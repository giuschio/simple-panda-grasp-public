import argparse

from panda_py import Desk, libfranka


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2")
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default="987654321")
    parser.add_argument("--platform", default="panda")
    parser.add_argument("--open-width", type=float, default=0.08)
    parser.add_argument("--closed-width", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--close-threshold", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    desk = Desk(args.host, args.user, args.password, platform=args.platform)
    if not desk.has_control():
        desk.take_control()
    desk.unlock()
    desk.activate_fci()

    gripper = libfranka.Gripper(args.host)
    gripper.move(width=args.open_width, speed=args.speed)
    open_width = gripper.read_once().width
    gripper.move(width=args.closed_width, speed=args.speed)
    closed_width = gripper.read_once().width

    print(f"Gripper width: open={open_width:.4f} m, closed={closed_width:.4f} m")
    if closed_width > args.close_threshold or closed_width >= open_width:
        raise RuntimeError("Gripper close test failed")


if __name__ == "__main__":
    main()
