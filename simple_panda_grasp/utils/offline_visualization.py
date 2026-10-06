import open3d as o3d
import numpy as np


def create_cylinder_between(p1, p2, radius=0.005, color=[1, 0, 0]):
    """Create a cylinder between two points."""
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=np.linalg.norm(p2 - p1))
    cylinder.compute_vertex_normals()
    
    # Align cylinder with the vector from p1 to p2
    direction = (p2 - p1)
    mid_point = (p1 + p2) / 2
    direction_norm = direction / np.linalg.norm(direction)

    # Create rotation matrix
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction_norm)
    if np.linalg.norm(axis) < 1e-6:
        R = np.eye(3)
    else:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cylinder.rotate(R, center=np.zeros(3))
    cylinder.translate(mid_point)
    cylinder.paint_uniform_color(color)
    
    return cylinder


import matplotlib.colors as mcolors
def get_colormap():
    # Define the colors: Red -> Yellow -> Green
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # RGB for Red, Yellow, Green
    n_bins = 100  # Number of discrete steps in the colormap

    # Create the colormap
    cmap_name = 'red_yellow_green'
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    def colormap(x):
        c = cm(x)
        return list(c[:3])
    return colormap


import hashlib
def string_to_color(s):
    # Create a hash of the string
    if s == "unclassified": return (0, 0, 0)
    hash_object = hashlib.md5(s.encode())
    hash_bytes = hash_object.digest()
    # Use the first 3 bytes of the hash as RGB values
    r, g, b = hash_bytes[0], hash_bytes[1], hash_bytes[2]
    return (r/255.0, g/255.0, b/255.0)


def visualize_scene_pcd_util(
    pcd,
    grasps,
    visualize_ref_system=True,
    color_key="success",
    visualize_grasp_rotation=False,
):
    """
    Visualize a point cloud and grasp markers in Open3D.

    Args:
        pcd: Open3D point cloud.
        grasps: List of dicts, each with 'pose', 'point1', 'point2', and optional 'success'.
        color_map: Function mapping success (0-1) -> color [r, g, b]. If None, default color is red.
        radius: Radius of the cylinders.
        visualize_grasp_rotation: If True, show an xyz reference frame at each grasp pose.
    """
    if color_key == "success" or "score" in color_key:
        color_map = get_colormap()
    elif color_key == "type":
        color_map = string_to_color
    else:
        raise ValueError(f"Unknown color_key: {color_key}")

    radius=0.001
    geometries = [pcd]

    for grasp in grasps:
        pose = grasp["pose"]
        p1 = np.array(grasp["point1"])
        p2 = np.array(grasp["point2"])
        z_dir = pose[:3, 2]

        p1_end = p1 - z_dir * 0.05
        p2_end = p2 - z_dir * 0.05
        color = color_map(grasp.get(color_key, 1.0))  # Default success to 1.0 if not present

        side1 = create_cylinder_between(p1, p1_end, radius, color)
        side2 = create_cylinder_between(p2, p2_end, radius, color)
        side3 = create_cylinder_between(0.5*(p2_end+p1_end), 0.5*(p2_end+p1_end) - z_dir * 0.05, radius, color)
        bar = create_cylinder_between(p1_end, p2_end, radius, color)

        geometries.extend([side1, side2, side3, bar])

        if visualize_grasp_rotation:
            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            grasp_frame.transform(pose)
            geometries.append(grasp_frame)

    if visualize_ref_system:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        geometries.append(frame)

    return geometries


def visualize_scene_pcd(
    pcd,
    grasps,
    visualize_ref_system=True,
    color_key="success",
    visualize_grasp_rotation=False,
):
    geometries = visualize_scene_pcd_util(
        pcd,
        grasps,
        visualize_ref_system=visualize_ref_system,
        color_key=color_key,
        visualize_grasp_rotation=visualize_grasp_rotation,
    )
    o3d.visualization.draw_geometries(geometries)
