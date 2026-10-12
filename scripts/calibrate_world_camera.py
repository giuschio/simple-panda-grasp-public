"""Register the Panda hand in one RGB-D image to calibrate a fixed camera."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
from omegaconf import OmegaConf
from pycpd import RigidRegistration

from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.utils.transform import Rotation, Transform


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/config_panda.yaml"
HAND_URDF = ROOT / "assets/robots/panda/hand.urdf"
VOXEL_SIZE = 0.004
HAND_DEPTH = 0.18
MAX_DEPTH = 1.5
FINGER_OPENING = 0.04
# The standard Franka EE/TCP is 103.4 mm along +Z from the panda_hand URDF origin.
T_HAND2EE = Transform(Rotation.identity(), [0.0, 0.0, -0.1034])


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
    cloud = model.sample_points_uniformly(30000).voxel_down_sample(VOXEL_SIZE)
    cloud.paint_uniform_color([0.9, 0.15, 0.1])
    return cloud


def capture_cloud(camera) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
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

    near = np.quantile(depth[valid], 0.05)
    foreground = valid & (depth < near + HAND_DEPTH)
    hand = o3d.geometry.PointCloud()
    hand.points = o3d.utility.Vector3dVector(points[foreground])
    hand = hand.voxel_down_sample(VOXEL_SIZE)
    labels = np.asarray(hand.cluster_dbscan(0.012, 12, print_progress=False))
    if labels.size and labels.max() >= 0:
        counts = np.bincount(labels[labels >= 0])
        hand = hand.select_by_index(np.flatnonzero(labels == counts.argmax()))
    if len(hand.points) < 100:
        raise RuntimeError("Could not isolate the hand; move it closer and keep it centered.")
    return scene, hand


def features(cloud: o3d.geometry.PointCloud):
    cloud = deepcopy(cloud)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.02, 40))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(0.05, 100)
    )
    return cloud, feature


def choose_side(matrix: np.ndarray, side: str) -> np.ndarray:
    camera_in_hand = np.linalg.inv(matrix)[:3, 3]
    wanted_sign = 1.0 if side == "front" else -1.0
    if np.sign(camera_in_hand[0]) != wanted_sign:
        symmetry = np.eye(4)
        symmetry[:3, :3] = Rotation.from_euler("z", np.pi).as_matrix()
        matrix = matrix @ symmetry
    return matrix


def register(model, observed) -> Transform:
    while True:
        side = input(
            "Camera sees the hand's front (+X/palm side) or back (-X)? "
            "[front/back]: "
        ).strip().lower()
        if side in ("front", "back"):
            break

    source, source_feature = features(model)
    target, target_feature = features(observed)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feature, target_feature, True, 3 * VOXEL_SIZE,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(3 * VOXEL_SIZE)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    initial = choose_side(result.transformation, side)

    rng = np.random.default_rng(0)
    x = np.asarray(observed.points)
    y = np.asarray(model.points)
    x = x[rng.choice(len(x), min(1800, len(x)), replace=False)]
    y = y[rng.choice(len(y), min(1800, len(y)), replace=False)]
    y_initial = y @ initial[:3, :3].T + initial[:3, 3]
    _, (_, rotation, translation) = RigidRegistration(
        X=x, Y=y_initial, scale=False, max_iterations=60, tolerance=1e-5, w=0.1
    ).register()
    delta = np.eye(4)
    delta[:3, :3] = rotation.T
    delta[:3, 3] = translation
    refined = o3d.pipelines.registration.registration_icp(
        model, observed, 0.015, delta @ initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    print(f"Registration fitness: {refined.fitness:.3f}, RMSE: {refined.inlier_rmse * 1000:.1f} mm")
    return Transform.from_matrix(refined.transformation)


def frame(transform: Transform, size: float):
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    axes.transform(transform.as_matrix())
    return axes


def main() -> None:
    panda = PandaGraspInterface(str(CONFIG))
    try:
        panda.open_gripper()
        print("Place the hand close to and centered in the fixed camera, then press Enter.")
        input()
        model = hand_cloud()
        scene, observed = capture_cloud(panda.camera)
        ee_to_base = panda.get_ee2panda()
        hand_to_camera = register(model, observed)
        hand_to_base = ee_to_base * T_HAND2EE
        camera_to_base = hand_to_base * hand_to_camera.inverse()
        registered = deepcopy(model).transform(hand_to_camera.as_matrix())
        ee_to_camera = hand_to_camera * T_HAND2EE.inverse()
        base_to_camera = camera_to_base.inverse()
        o3d.visualization.draw_geometries([
            scene, registered, frame(ee_to_camera, 0.07), frame(base_to_camera, 0.14)
        ], window_name="Camera cloud + hand (EE frame small, base frame large)")

        input("Move the hand to a different pose, then press Enter to validate: ")
        validation_scene, _ = capture_cloud(panda.camera)
        ee_to_base = panda.get_ee2panda()
        ee_to_camera = camera_to_base.inverse() * ee_to_base
        predicted_hand_to_camera = ee_to_camera * T_HAND2EE
        predicted_hand = deepcopy(model).transform(predicted_hand_to_camera.as_matrix())
        o3d.visualization.draw_geometries([
            validation_scene, predicted_hand, frame(ee_to_camera, 0.07),
            frame(camera_to_base.inverse(), 0.14),
        ], window_name="Validation (EE frame small, base frame large)")

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
