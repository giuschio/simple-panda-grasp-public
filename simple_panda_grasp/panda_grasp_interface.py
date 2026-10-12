import numpy as np
from copy import deepcopy
from random import choice
from omegaconf import OmegaConf

from simple_panda_grasp.utils.transform import Rotation, Transform
from realsense_utils2.scene_camera import SceneCameraCDM, SceneCameraFDM, SceneCameraRaw

from simple_panda_grasp.panda_constants import RELEASE_JOINTS_WIDE, SCANNING_JOINTS, SCANNING_JOINTS_DEFAULT, SCANNING_JOINTS_RANDOM, RELEASE_JOINTS_FRONT
from panda_py import Panda
from panda_py import libfranka
from panda_py import ik, ik_full

from time import sleep

from simple_panda_grasp.utils.dataclasses import CameraIntrinsic
import readchar


WORKSPACE_SIZE = 0.30
VIEWPOINT_ORIGIN = Transform(
    Rotation.identity(),
    np.r_[WORKSPACE_SIZE / 2, WORKSPACE_SIZE / 2, 0.10],
)


def ik2(pose: np.ndarray, q_init=None, n_q7: int = 200, safety_deg: float = 10.0):
    """Sweep q7, collect IK solutions, keep those within safety margin, pick closest to q_start."""
    joint_limits = np.array(
        [
            [-2.8973,  2.8973],  # joint1
            [-1.7628,  1.7628],  # joint2
            [-2.8973,  2.8973],  # joint3
            [-3.0718, -0.0698],  # joint4
            [-2.8973,  2.8973],  # joint5
            [-0.0175,  3.7525],  # joint6
            [-2.8973,  2.8973],  # joint7
        ],
        dtype=float,
    )

    lower, upper = joint_limits[:, 0], joint_limits[:, 1]

    q_start = np.array(
        [0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633, 0.78539816],
        dtype=float,
    )
    if q_init is None:
        q_init = q_start.copy()

    q7_values = np.linspace(lower[6], upper[6], n_q7)

    def _as_solutions(x) -> np.ndarray:
        # ik_full returns 4 solutions; this flattens (4,7) / (1,4,7) / etc. into (N,7)
        return np.asarray(x, dtype=float).reshape(-1, 7)

    # 1) Gather all finite candidates across the q7 sweep
    candidates = []
    for q7 in q7_values:
        sols = _as_solutions(ik_full(pose, q_init=q_init, q_7=q7))
        candidates.extend(sols[np.isfinite(sols).all(axis=1)])

    # 2) Apply safety margin
    safety_rad = np.deg2rad(safety_deg)
    lower_safe = lower + safety_rad
    upper_safe = upper - safety_rad

    best_joint = None
    if candidates:
        joints = np.asarray(candidates, dtype=float).reshape(-1, 7)
        within_safe = np.logical_and(joints >= lower_safe, joints <= upper_safe).all(axis=1)

        if np.any(within_safe):
            joints = joints[within_safe]

            # 3) Pick closest to the seed (mean absolute deviation)
            diffs = np.mean(np.abs(joints - q_init), axis=1)
            best_joint = joints[int(np.argmin(diffs))]

    return best_joint if best_joint is not None else np.full(7, np.nan)


def sample_world2cam() -> Transform:
    r = np.random.uniform(0.4, 0.5)
    theta = np.random.uniform(np.pi / 4, np.pi / 3)

    width = np.deg2rad(45)
    center = np.deg2rad(270)
    phi = np.random.uniform(center-width, center+width)
    eye = np.r_[
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ]
    return Transform.look_at(eye, np.zeros(3), np.r_[0.0, 0.0, 1.0]) * VIEWPOINT_ORIGIN.inverse()


def sample_reachable_camera2world(
    T_camera2ee: Transform,
    T_world2panda: Transform,
    max_attempts: int = 100,
) -> tuple[Transform, np.ndarray]:
    for _ in range(max_attempts):
        T_camera2world = sample_world2cam().inverse()
        T_ee2world = T_camera2world * T_camera2ee.inverse()
        T_ee2panda = T_world2panda * T_ee2world
        joints = ik2(T_ee2panda.as_matrix())
        if np.isfinite(joints).all():
            return T_camera2world, joints
    raise RuntimeError(f"Failed to sample a reachable camera viewpoint after {max_attempts} attempts")

class SceneCameraFDMHD(SceneCameraFDM):
    def __init__(self):
        super().__init__(fdm_scale=1.0)


class PandaGraspInterface:
    def __init__(
        self,
        config_path: str,
    ):
        config = OmegaConf.load(config_path)

        hostname = config["hostname"]
        camera_type = config.get("camera_type", "camera_raw")
        T_world2panda = Transform.from_list(config.get("T_world2panda", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        T_camera2ee = Transform.from_list(config.get("T_camera2ee", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        camera_classes = {
            "camera_raw": SceneCameraRaw,
            "camera_cdm": SceneCameraCDM,
            "camera_fdm": SceneCameraFDMHD,
        }
        if camera_type not in camera_classes:
            raise ValueError(f"Unknown camera_type: {camera_type}")
        self.observation_viewpoints = {
            "default": SCANNING_JOINTS_DEFAULT[0],
            "top_down": [-0.106, -0.672, 0.011, -1.427, 0.016, 1.054, 0.698],
            "scan": SCANNING_JOINTS,
        }
        viewpoint_config = config.get("observation_viewpoints", OmegaConf.create({}))
        self.observation_viewpoints.update(OmegaConf.to_container(viewpoint_config))
        self.camera = camera_classes[camera_type]()
        self.robot = Panda(hostname)
        self.gripper = libfranka.Gripper(hostname)
        self._last_pick_grasp = None

        self.open_gripper()
        self.workspace_size = 0.50
        self.z_max = 0.40

        offset = (0.30 - self.workspace_size) / 2
        self.T_world2panda = T_world2panda * Transform(Rotation.identity(), [offset, offset, 0.0])
        self.T_camera2ee = T_camera2ee
        
        lx, ux = 0.0, self.workspace_size
        ly, uy = 0.0, self.workspace_size
        lz, uz = -0.02, self.z_max
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

        self.world_z_coordinate_safety_offset = 0.05
        self.joint_movement_speed_factor = 0.5

    def reset(self, seed=0, *args, **kwargs):
        self._last_pick_grasp = None
        # self.move_to_start()
        print("Human, please reset the scene if necessary")
        print("Press Enter to continue, or any other key to interrupt")
        key = readchar.readkey()

        if key in ('\r', '\n'):
            print("Enter pressed")
            return True
        else:
            print("Aborted by user")
            return False

    @property
    def T_panda2world(self):
        return self.T_world2panda.inverse()

    def move_ee(self, target_pose: Transform, use_ik=False) -> bool:
        T_ee2panda = self.T_world2panda * target_pose
        if use_ik is False:
            return self.robot.move_to_pose(position=T_ee2panda.translation, orientation=T_ee2panda.rotation.as_quat())
        else:
            joint_positions = ik2(T_ee2panda.as_matrix())
            if np.isnan(joint_positions).any():
                print("Could not find ik2 solution for desired pose. Maybe try with use_ik=False?")
                return False
            else:
                return self.move_jj(joint_positions)

    def move_ee_4d(self, target_position: np.ndarray, target_rotation: float, use_ik=False) -> bool:
        r = Rotation.from_euler('z', target_rotation, degrees=True) * Rotation.from_euler('xz', [180, 90], degrees=True)
        self.move_ee(Transform(r, target_position), use_ik=use_ik)

    def move_jj(self, target_joints: np.ndarray, speed_factor=None) -> bool:
        joints = np.asarray(target_joints, dtype=np.float64)
        if speed_factor is None: speed_factor = self.joint_movement_speed_factor
        return self.robot.move_to_joint_position(positions=joints, speed_factor=speed_factor)
    
    def move_to_start(self):
        return self.robot.move_to_start(speed_factor=self.joint_movement_speed_factor)
    
    def get_jj(self) -> np.ndarray:
        return np.asarray(self.robot.get_robot().read_once().q)
    
    def print_jj(self):
        formatted = ", ".join(f"{jp:.3f}" for jp in self.get_jj())
        print(f"[{formatted}]")

    def move_camera(self, target_pose: Transform, use_ik=False) -> bool:
        T_camera2world = target_pose
        T_ee2world = T_camera2world * self.T_camera2ee.inverse()
        return self.move_ee(T_ee2world, use_ik=use_ik)
    
    def get_ee2panda(self) -> Transform:
        state = self.robot.get_robot().read_once()
        matrix = np.asarray(state.O_T_EE).reshape(4, 4, order="F")
        return Transform.from_matrix(matrix)

    def get_ee2world(self) -> Transform:
        T_ee2panda = self.get_ee2panda()
        return self.T_panda2world * T_ee2panda
    
    def get_cam2panda(self) -> Transform:
        T_ee2panda = self.get_ee2panda()
        return T_ee2panda * self.T_camera2ee
    
    def get_cam2world(self) -> Transform:
        T_ee2world = self.get_ee2world()
        return T_ee2world * self.T_camera2ee

    def close_gripper(self):
        self.gripper.grasp(
                width=0.001,
                speed=0.03,
                force=40.0,
                epsilon_inner=0.001,
                epsilon_outer=0.07,
            )
        state = self.gripper.read_once()
        return state.width

    def open_gripper(self):
        self.gripper.move(
            width=0.08,     # e.g. 8 cm open
            speed=0.1
        )
        state = self.gripper.read_once()
        return state.width
    
    def get_gripper_width(self):
        return self.gripper.read_once().width

    def get_picture(self):
        color, depth = self.camera.capture()
        return color, depth
    
    def _get_grasp_waypoints_util(self, grasp: dict, refine_pose: bool = False):
        T_world_grasp = Transform.from_matrix(grasp["pose"])
        if refine_pose and T_world_grasp.translation[2] < -self.world_z_coordinate_safety_offset:
            T_world_grasp.translation[3] += (T_world_grasp.translation[3]-self.world_z_coordinate_safety_offset)
        
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        lift = T_world_grasp.translation[2] + self.world_z_coordinate_safety_offset
        T_world_retreat = Transform(Rotation.identity(), [0.0, 0.0, lift]) * T_world_grasp

        grasp["pose_pre"] = T_world_pregrasp.as_matrix()
        grasp["pose_post"] = T_world_retreat.as_matrix()

        q_start = np.array([0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079633, 0.78539816])
        grasp["pose_jj"] = ik2(self.T_world2panda.as_matrix() @ grasp["pose"], q_start)
        if not np.isfinite(grasp["pose_jj"]).all():
            return None
        grasp["pose_pre_jj"] = ik2(self.T_world2panda.as_matrix() @ grasp["pose_pre"], grasp["pose_jj"])
        grasp["pose_post_jj"] = ik2(self.T_world2panda.as_matrix() @ grasp["pose_post"], grasp["pose_jj"])

        joints = np.array([grasp[k] for k in ("pose_pre_jj", "pose_jj", "pose_post_jj")])
        if not np.isfinite(joints).all() or np.max(np.abs(np.diff(joints, axis=0))) > np.pi / 2:
            return None
        return grasp
    
    def run_grasp_checks(self, grasps: list, refine_pose=False):
        grasps = [
            wp
            for g in grasps
            if (wp := self._get_grasp_waypoints_util(g, refine_pose)) is not None
        ]
        return grasps

    def _ask_recover_and_continue_after_motion_failure(self, motion_name: str) -> bool:
        print(f"{motion_name} failed. Clear robot error and continue as if this motion succeeded (y/n)? ")
        key = readchar.readkey().lower()
        print(key)
        if key != "y":
            return False

        try:
            self.robot.recover()
        except Exception as exc:
            print(f"Automatic error recovery failed: {exc}")
            return False
        return True
    
    def pick(self, grasp: dict):
        self._last_pick_grasp = None
        self.open_gripper()
        res = self.move_jj(grasp["pose_pre_jj"])
        if res is False and not self._ask_recover_and_continue_after_motion_failure("T_world_pregrasp"):
            return False, "Failed to move to T_world_pregrasp"
        print("T_world_pregrasp success")

        res = self.move_jj(grasp["pose_jj"], 0.1)
        if res is False and not self._ask_recover_and_continue_after_motion_failure("T_world_grasp"):
            return False, "Failed to move to T_world_grasp"
        print("T_world_grasp success")

        _ = self.close_gripper()

        res = self.move_jj(grasp["pose_post_jj"], 0.1)
        if res is False and not self._ask_recover_and_continue_after_motion_failure("T_world_retreat"):
            return False, "Failed to move to T_world_retreat"
        print("T_world_retreat success")

        state = self.gripper.read_once()
        if not state.is_grasped:
            return False, "Failed to grasp"
        self._last_pick_grasp = deepcopy(grasp)
        return True, "Grasp success"
    
    def place(self, position="side"):
        if position == "front":
            return self.place_front()
        return self.place_side()
    
    def place_side(self):
        self.move_jj(RELEASE_JOINTS_WIDE)
        self.open_gripper()
        self._last_pick_grasp = None
        return True
    
    def place_front(self):
        t = self.get_ee2world()
        t.translation += np.array([0., 0., 0.2])
        self.move_ee(t)
        self.move_jj(RELEASE_JOINTS_FRONT)
        self.open_gripper()
        self._last_pick_grasp = None
        return True
    
    def place_reset(self):
        self.open_gripper()
        self._last_pick_grasp = None
        return True
    
    def place_back(self):
        if self._last_pick_grasp is None:
            return False

        grasp = self._last_pick_grasp
        res = self.move_jj(grasp["pose_jj"], 0.1)
        if res is False: return False

        self.open_gripper()

        res = self.move_jj(grasp["pose_pre_jj"], 0.1)
        if res is False: return False

        self._last_pick_grasp = None
        return True
    
    def capture_observation(self, viewpoint=None) -> dict:
        """Capture one RGB-D observation, optionally moving to a named viewpoint.

        None captures at the current pose. Named viewpoints are ``default``,
        ``top_down``, and ``sampled`` (chosen from SCANNING_JOINTS_RANDOM).
        ``multiview`` captures the scan sequence.
        Returns the existing batched dictionary: RGB uint8, depth float32 in
        metres, intrinsics, world-to-camera extrinsics (xyzw + xyz), and eyes
        in world coordinates. No reconstruction or filtering is performed.
        """
        if viewpoint == "multiview":
            return self.capture_observations("scan")
        if viewpoint is not None:
            if viewpoint == "sampled":
                joints = choice(SCANNING_JOINTS_RANDOM)
            else:
                if viewpoint == "scan" or viewpoint not in self.observation_viewpoints:
                    raise ValueError(f"Unknown single viewpoint: {viewpoint!r}")
                joints = self.observation_viewpoints[viewpoint]
            return self._capture_at_joints(joints)
        cam2world = self.get_cam2world()
        color, depth = self.get_picture()
        k = self.camera.color_intrinsics
        return {
            "depth_imgs": np.asarray(depth, dtype=np.float32)[None].copy(),
            "rgb_imgs": np.ascontiguousarray(color[..., ::-1])[None],
            "intrinsics": CameraIntrinsic(
                width=k.width, height=k.height, fx=k.fx, fy=k.fy, cx=k.cx, cy=k.cy,
            ),
            "extrinsics": np.asarray([cam2world.inverse().to_list()], dtype=np.float64),
            "eye": np.asarray([cam2world.translation], dtype=np.float64),
        }

    def _capture_at_joints(self, joints) -> dict:
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("An observation viewpoint must contain seven finite joint angles")
        if not self.move_jj(joints):
            raise RuntimeError("Failed to move to observation viewpoint")
        sleep(1.0)
        return self.capture_observation()

    def capture_observations(self, viewpoints="scan") -> dict:
        """Capture a configured scan or a sequence of named single viewpoints."""
        if isinstance(viewpoints, str):
            if viewpoints not in ("scan", "multiview"):
                return self.capture_observation(viewpoints)
            observations = [self._capture_at_joints(q) for q in self.observation_viewpoints["scan"]]
        else:
            observations = [self.capture_observation(name) for name in viewpoints]
        if not observations:
            raise ValueError("At least one viewpoint is required")
        result = {"intrinsics": observations[0]["intrinsics"]}
        for key in ("depth_imgs", "rgb_imgs", "extrinsics", "eye"):
            result[key] = np.concatenate([obs[key] for obs in observations], axis=0)
        return result

    def get_observation_images_fixed_viewpoints(self) -> dict:
        """Compatibility alias for the configured scan."""
        return self.capture_observations("scan")

    def get_observation_random_viewpoint(self) -> dict:
        """Compatibility alias for sampling from SCANNING_JOINTS_RANDOM."""
        return self.capture_observation("sampled")

    def get_observation_default_viewpoint(self) -> dict:
        """Compatibility alias for the fixed default observation viewpoint."""
        return self.capture_observation("default")

    def get_observation_top_down(self) -> dict:
        return self.capture_observation("top_down")
