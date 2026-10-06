import numpy as np
import open3d as o3d


def visualize_grasps(cloud, grasps):
    geometries = [cloud]
    for grasp in grasps:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
        frame.transform(grasp["pose"])
        contacts = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([grasp["point1"], grasp["point2"]]),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        contacts.paint_uniform_color([1, 0, 0])
        geometries.extend([frame, contacts])
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Reachable grasp proposals",
    )


class EdgeGraspSampler:
    """Top-down grasp sampler aligned with local surface normals."""

    def __init__(self, seed=None, top_layer=0.04, max_width=0.08, max_normal_z=0.5):
        self.rng = np.random.default_rng(seed)
        self.top_layer = top_layer
        self.max_width = max_width
        self.max_normal_z = max_normal_z

    def __call__(self, cloud_objects, n_grasps=1):
        if not cloud_objects.has_normals():
            cloud_objects.estimate_normals()
            cloud_objects.orient_normals_towards_camera_location(np.array([0.15, 0.15, 0.40]))

        points = np.asarray(cloud_objects.points, dtype=np.float32)
        normals = np.asarray(cloud_objects.normals, dtype=np.float32)

        valid = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(normals).all(axis=1)
            & (points[:, 2] >= points[:, 2].max() - self.top_layer)
            & (np.abs(normals[:, 2]) <= self.max_normal_z)
        )
        points, normals = points[valid], normals[valid]
        if len(points) == 0:
            return []

        grasps = []
        for _ in range(100 * n_grasps):
            if len(grasps) >= n_grasps:
                break
            grasp = self._sample_single_grasp(points, normals)
            if grasp is not None:
                grasps.append(grasp)
        return grasps

    def _sample_single_grasp(self, points, normals):
        idx = self.rng.integers(len(points))
        contact = points[idx]

        y_axis = normals[idx].copy()
        y_axis[2] = 0.0
        norm = np.linalg.norm(y_axis)
        if norm < 1e-6:
            return None
        y_axis /= norm

        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        x_axis = np.cross(y_axis, z_axis)

        # Put the gripper center half a jaw width inward from the contact.
        center = contact - 0.5 * self.max_width * y_axis + 0.01 * y_axis

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        pose[:3, 3] = center + 0.01 * z_axis

        point1 = contact
        point2 = contact - self.max_width * y_axis + 0.01 * y_axis

        return dict(
            pose=pose,
            width=self.max_width,
            point1=point1,
            point2=point2,
            sampler="edge",
        )