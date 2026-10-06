import numpy as np
import open3d as o3d


class ConservativeVoxelMap:
    """
    Conservative occupancy map for one cleaned point cloud.

    unsafe == True:
        - occupied
        - behind an observed surface

    unsafe == False:
        - observed free space or rays with no hit
    """

    def __init__(self, bbox_min, bbox_max, resolution):
        self.bbox_min = np.asarray(bbox_min, dtype=float)
        self.bbox_max = np.asarray(bbox_max, dtype=float)
        self.resolution = float(resolution)

        extent = self.bbox_max - self.bbox_min
        self.shape = np.ceil(extent / self.resolution).astype(int)

        # Start optimistic: rays with no hit stay safe.
        self.unsafe = np.zeros(self.shape, dtype=bool)

    def __call__(self, pointcloud, eye):
        """
        pointcloud:
            Open3D point cloud with cleaned world-frame surface points.

        eye:
            (3,) sensor origin in world frame.
        """
        points = np.asarray(pointcloud.points, dtype=float)
        eye = np.asarray(eye, dtype=float)
        self.unsafe.fill(False)

        if len(points) > 0:
            self._mark_unsafe_behind_hits(points, eye)

        unsafe_pointcloud = o3d.geometry.PointCloud()
        unsafe_idxs = np.argwhere(self.unsafe)
        if len(unsafe_idxs) > 0:
            unsafe_pointcloud.points = o3d.utility.Vector3dVector(self._idx_to_world(unsafe_idxs))
        return unsafe_pointcloud

    def _world_to_idx(self, points):
        points = np.asarray(points, dtype=float)
        return np.floor((points - self.bbox_min) / self.resolution).astype(int)

    def _idx_to_world(self, idx):
        idx = np.asarray(idx, dtype=float)
        return self.bbox_min + (idx + 0.5) * self.resolution

    def _in_bounds_idx(self, idx):
        idx = np.asarray(idx)
        return np.all((idx >= 0) & (idx < self.shape), axis=-1)

    def _mark_unsafe_behind_hits(self, hits, eye):
        directions = hits - eye[None, :]
        hit_distances = np.linalg.norm(directions, axis=1)
        valid = hit_distances > 1e-9
        if not np.any(valid):
            return

        hits = hits[valid]
        directions = directions[valid] / hit_distances[valid, None]
        hit_distances = hit_distances[valid]

        hit_idxs = self._world_to_idx(hits)
        in_bounds_hits = self._in_bounds_idx(hit_idxs)
        hit_idxs = hit_idxs[in_bounds_hits]
        if len(hit_idxs) > 0:
            self.unsafe[
                hit_idxs[:, 0],
                hit_idxs[:, 1],
                hit_idxs[:, 2],
            ] = True

        t_exit = self._ray_box_exit_distances(directions, eye)
        valid_exit = np.isfinite(t_exit) & (t_exit > hit_distances)
        if not np.any(valid_exit):
            return

        directions = directions[valid_exit]
        hit_distances = hit_distances[valid_exit]
        t_exit = t_exit[valid_exit]

        step = self.resolution * 0.5
        chunk_size = 2048
        for start in range(0, len(directions), chunk_size):
            end = min(start + chunk_size, len(directions))
            chunk_dirs = directions[start:end]
            chunk_hit = hit_distances[start:end]
            chunk_exit = t_exit[start:end]
            max_steps = int(np.ceil(np.max((chunk_exit - chunk_hit) / step))) + 1
            distances = chunk_hit[:, None] + np.arange(max_steps)[None, :] * step
            behind_surface = distances <= chunk_exit[:, None]
            samples = eye[None, None, :] + distances[:, :, None] * chunk_dirs[:, None, :]
            idxs = self._world_to_idx(samples[behind_surface])
            idxs = idxs[self._in_bounds_idx(idxs)]
            if len(idxs) > 0:
                self.unsafe[
                    idxs[:, 0],
                    idxs[:, 1],
                    idxs[:, 2],
                ] = True

    def _ray_box_exit_distances(self, directions, eye):
        t_min = np.full(len(directions), -np.inf, dtype=float)
        t_max = np.full(len(directions), np.inf, dtype=float)
        valid = np.ones(len(directions), dtype=bool)

        for axis in range(3):
            axis_dir = directions[:, axis]
            parallel = np.abs(axis_dir) < 1e-12
            outside_parallel = parallel & (
                (eye[axis] < self.bbox_min[axis]) | (eye[axis] > self.bbox_max[axis])
            )
            valid &= ~outside_parallel

            non_parallel = ~parallel
            t0 = np.empty(len(directions), dtype=float)
            t1 = np.empty(len(directions), dtype=float)
            t0[non_parallel] = (self.bbox_min[axis] - eye[axis]) / axis_dir[non_parallel]
            t1[non_parallel] = (self.bbox_max[axis] - eye[axis]) / axis_dir[non_parallel]
            axis_t_min = np.minimum(t0[non_parallel], t1[non_parallel])
            axis_t_max = np.maximum(t0[non_parallel], t1[non_parallel])
            t_min[non_parallel] = np.maximum(t_min[non_parallel], axis_t_min)
            t_max[non_parallel] = np.minimum(t_max[non_parallel], axis_t_max)

        valid &= t_max >= np.maximum(t_min, 0.0)
        return np.where(valid, t_max, -np.inf)
