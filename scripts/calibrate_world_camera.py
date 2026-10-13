"""Register the Panda hand in one RGB-D image to calibrate a fixed camera."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.utils.transform import Rotation, Transform


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/config_panda.yaml"
HAND_URDF = ROOT / "assets/robots/panda/hand.urdf"
VOXEL_SIZE = 0.004
MAX_DEPTH = 1.5
FINGER_OPENING = 0.04
N_CALIBRATION_POSES = 3
N_BOUNDING_POINTS = 4
BOUNDING_BOX_PADDING = np.array([0.035, 0.035, 0.050])
MIN_HAND_POINTS = 200
NORMAL_RADIUS = 4 * VOXEL_SIZE
FPFH_RADIUS = 5 * VOXEL_SIZE
GLOBAL_DISTANCE = 4 * VOXEL_SIZE
COARSE_NORMAL_DISTANCE = 0.040
FINE_NORMAL_DISTANCE = 0.015
# panda_hand is fixed to the Franka flange with a -45 degree rotation about Z.
# Using the flange makes calibration independent of the EE/TCP configured in Desk.
T_HAND2FLANGE = Transform(
    Rotation.from_euler("z", -np.pi / 4), np.zeros(3)
)


def pose(xyz="0 0 0", rpy="0 0 0") -> Transform:
    return Transform(
        Rotation.from_euler("xyz", np.fromstring(rpy, sep=" ")),
        np.fromstring(xyz, sep=" "),
    )


def hand_cloud() -> o3d.geometry.PointCloud:
    """Sample the URDF visual meshes in the panda_hand frame."""
    root = ET.parse(HAND_URDF).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    meshes = []

    for name, link in links.items():
        visual = link.find("visual")
        if visual is None:
            continue
        mesh_tag = visual.find("geometry/mesh")
        mesh = o3d.io.read_triangle_mesh(str(HAND_URDF.parent / mesh_tag.get("filename")))
        origin = visual.find("origin")
        T_mesh2link = pose(**origin.attrib) if origin is not None else Transform.identity()
        T_link2hand = Transform.identity()
        child = name
        while child in joints:
            joint = joints[child]
            origin = joint.find("origin")
            T_child2parent = pose(**origin.attrib) if origin is not None else Transform.identity()
            if joint.get("type") == "prismatic":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                T_child2parent.translation += axis * FINGER_OPENING
            T_link2hand = T_child2parent * T_link2hand
            child = joint.find("parent").get("link")
        mesh.transform((T_link2hand * T_mesh2link).as_matrix())
        meshes.append(mesh)

    model = meshes[0]
    for mesh in meshes[1:]:
        model += mesh
    model.compute_vertex_normals()
    cloud = model.sample_points_uniformly(30000).voxel_down_sample(VOXEL_SIZE)
    cloud.normalize_normals()
    cloud.paint_uniform_color([0.9, 0.15, 0.1])
    return cloud


def capture_cloud(camera) -> o3d.geometry.PointCloud:
    color, depth = camera.capture()
    depth = np.asarray(depth, dtype=float)
    color = np.asarray(color)[..., ::-1]
    k = camera.color_intrinsics
    ys, xs = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0) & (depth < MAX_DEPTH)
    points = np.stack(((xs - k.cx) * depth / k.fx,
                       (ys - k.cy) * depth / k.fy, depth), axis=-1)

    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(points[valid])
    scene.colors = o3d.utility.Vector3dVector(color[valid] / 255.0)
    scene = scene.voxel_down_sample(VOXEL_SIZE)
    return scene


def pick_hand(scene: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Crop and clean the hand using four points selected in the scene cloud."""
    while True:
        print(
            f"Select exactly {N_BOUNDING_POINTS} points around the hand with "
            "Shift + left click. Use Shift + right click to undo, then press Q."
        )
        viewer = o3d.visualization.VisualizerWithEditing()
        viewer.create_window(window_name="Select four hand bounding points")
        viewer.add_geometry(scene)
        viewer.get_render_option().point_size = 4.0
        viewer.run()
        viewer.destroy_window()
        picked = viewer.get_picked_points()
        if len(picked) != N_BOUNDING_POINTS:
            print(
                f"Selected {len(picked)} points; exactly "
                f"{N_BOUNDING_POINTS} are required."
            )
            continue

        picked_points = np.asarray(scene.points)[picked]
        bounds = o3d.geometry.AxisAlignedBoundingBox(
            picked_points.min(axis=0) - BOUNDING_BOX_PADDING,
            picked_points.max(axis=0) + BOUNDING_BOX_PADDING,
        )
        hand = scene.crop(bounds)
        if len(hand.points) < MIN_HAND_POINTS:
            print(
                f"The selected box contains only {len(hand.points)} points; "
                "select points farther apart around the hand."
            )
            continue

        hand, _ = hand.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        hand, _ = hand.remove_radius_outlier(nb_points=8, radius=3 * VOXEL_SIZE)
        if len(hand.points) < MIN_HAND_POINTS:
            print(
                f"Only {len(hand.points)} hand points remain after cleanup; "
                "select a tighter box with less background."
            )
            continue
        hand.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=NORMAL_RADIUS, max_nn=50
            )
        )
        hand.orient_normals_towards_camera_location(np.zeros(3))
        hand.normalize_normals()
        print(f"Using {len(hand.points)} cleaned hand points for registration.")
        return hand


def register(model, observed) -> Transform:
    o3d.utility.random.seed(0)
    source_features = o3d.pipelines.registration.compute_fpfh_feature(
        model,
        o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=100),
    )
    target_features = o3d.pipelines.registration.compute_fpfh_feature(
        observed,
        o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=100),
    )
    global_result = (
        o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            model,
            observed,
            source_features,
            target_features,
            True,
            GLOBAL_DISTANCE,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    GLOBAL_DISTANCE
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
                    np.deg2rad(45)
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
    )
    print(
        f"FPFH/RANSAC fitness: {global_result.fitness:.3f}, "
        f"RMSE: {global_result.inlier_rmse * 1000:.1f} mm"
    )

    normal_source = deepcopy(model).transform(global_result.transformation)
    normal_source.orient_normals_towards_camera_location(np.zeros(3))
    coarse_symmetric = o3d.pipelines.registration.TransformationEstimationSymmetric(
        o3d.pipelines.registration.HuberLoss(k=FINE_NORMAL_DISTANCE)
    )
    fine_symmetric = o3d.pipelines.registration.TransformationEstimationSymmetric(
        o3d.pipelines.registration.TukeyLoss(k=FINE_NORMAL_DISTANCE)
    )
    coarse = o3d.pipelines.registration.registration_symmetric_icp(
        normal_source,
        observed,
        COARSE_NORMAL_DISTANCE,
        np.eye(4),
        coarse_symmetric,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    fine = o3d.pipelines.registration.registration_symmetric_icp(
        normal_source,
        observed,
        FINE_NORMAL_DISTANCE,
        coarse.transformation,
        fine_symmetric,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    normal_transform = fine.transformation @ global_result.transformation
    print(
        f"Normal-aware fitness: {fine.fitness:.3f}, "
        f"RMSE: {fine.inlier_rmse * 1000:.1f} mm"
    )
    return Transform.from_matrix(normal_transform)


def frame(transform: Transform, size: float):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    axes.transform(transform.as_matrix())
    return axes


def robot_poses(panda: PandaGraspInterface) -> tuple[Transform, Transform]:
    """Return hand-to-base and configured-EE-to-base from one robot state."""
    state = panda.robot.get_robot().read_once()
    ee_to_base = Transform.from_matrix(
        np.asarray(state.O_T_EE).reshape(4, 4, order="F")
    )
    ee_to_flange = Transform.from_matrix(
        np.asarray(state.F_T_EE).reshape(4, 4, order="F")
    )
    flange_to_base = ee_to_base * ee_to_flange.inverse()
    hand_to_base = flange_to_base * T_HAND2FLANGE
    return hand_to_base, ee_to_base


def mean_transform(transforms: list[Transform]) -> Transform:
    rotations = Rotation.from_matrix(
        np.stack([transform.rotation.as_matrix() for transform in transforms])
    ).mean()
    translations = np.mean(
        [transform.translation for transform in transforms], axis=0
    )
    return Transform(rotations, translations)


def pose_error(reference: Transform, estimate: Transform) -> tuple[float, float]:
    translation = np.linalg.norm(reference.translation - estimate.translation)
    rotation = reference.rotation * estimate.rotation.inv()
    return float(translation), float(np.degrees(rotation.magnitude()))


def main() -> None:
    panda = PandaGraspInterface(str(CONFIG))
    try:
        # panda.open_gripper()
        model = hand_cloud()
        camera_to_base_estimates = []

        while len(camera_to_base_estimates) < N_CALIBRATION_POSES:
            index = len(camera_to_base_estimates) + 1
            instruction = "Place" if index == 1 else "Move"
            input(
                f"{instruction} the hand at calibration pose "
                f"{index}/{N_CALIBRATION_POSES}, close to and centered in the "
                "fixed camera. Use a distinctly different orientation for each "
                "pose, then press Enter: "
            )
            scene = capture_cloud(panda.camera)
            hand_to_base, ee_to_base = robot_poses(panda)
            observed = pick_hand(scene)
            hand_to_camera = register(model, observed)
            camera_to_base_estimate = hand_to_base * hand_to_camera.inverse()

            registered = deepcopy(model).transform(hand_to_camera.as_matrix())
            ee_to_camera = camera_to_base_estimate.inverse() * ee_to_base
            o3d.visualization.draw_geometries([
                scene,
                registered,
                frame(ee_to_camera, 0.07),
                frame(camera_to_base_estimate.inverse(), 0.14),
            ], window_name=(
                f"Calibration pose {index}/{N_CALIBRATION_POSES} "
                "(EE frame small, base frame large)"
            ))

            accepted = input(
                "Does this registration look correct? [y/N]: "
            ).strip().lower()
            if accepted == "y":
                camera_to_base_estimates.append(camera_to_base_estimate)
            else:
                print("Registration rejected; repeat this pose.")

        camera_to_base = mean_transform(camera_to_base_estimates)
        print("Calibration-pose disagreement from the combined estimate:")
        for index, estimate in enumerate(camera_to_base_estimates, start=1):
            translation_error, rotation_error = pose_error(camera_to_base, estimate)
            print(
                f"  Pose {index}: {translation_error * 1000:.1f} mm, "
                f"{rotation_error:.2f} deg"
            )

        input("Move the hand to a different pose, then press Enter to validate: ")
        validation_scene = capture_cloud(panda.camera)
        hand_to_base, ee_to_base = robot_poses(panda)
        validation_observed = pick_hand(validation_scene)
        predicted_hand_to_camera = camera_to_base.inverse() * hand_to_base
        predicted_hand = deepcopy(model).transform(predicted_hand_to_camera.as_matrix())

        print("Registering the validation pose independently to measure its error.")
        observed_hand_to_camera = register(model, validation_observed)
        translation_error, rotation_error = pose_error(
            observed_hand_to_camera, predicted_hand_to_camera
        )
        print(
            f"Validation error: {translation_error * 1000:.1f} mm, "
            f"{rotation_error:.2f} deg"
        )
        observed_hand = deepcopy(model)
        observed_hand.paint_uniform_color([0.1, 0.8, 0.2])
        observed_hand.transform(observed_hand_to_camera.as_matrix())
        ee_to_camera = camera_to_base.inverse() * ee_to_base
        o3d.visualization.draw_geometries([
            validation_scene, predicted_hand, observed_hand,
            frame(ee_to_camera, 0.07),
            frame(camera_to_base.inverse(), 0.14),
        ], window_name=(
            "Validation (prediction red, independent registration green)"
        ))

        if input("Does the validation overlay look correct? [y/N]: ").strip().lower() == "y":
            config = OmegaConf.load(CONFIG)
            config.T_camera2panda = camera_to_base.to_list()
            OmegaConf.save(config, CONFIG)
            print(f"Saved T_camera2panda to {CONFIG}")
        else:
            print("Calibration discarded.")
    finally:
        panda.camera.finalize()


if __name__ == "__main__":
    main()
