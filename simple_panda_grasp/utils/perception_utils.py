from typing import List, Optional


import numpy as np
import open3d as o3d

from simple_panda_grasp.utils.transform import Transform


class ScalableTSDFVolume:
    """Sparse tensor TSDF; allocation is independent of workspace bounds.

    Uses VoxelBlockGrid because Open3D 0.20's legacy ScalableTSDFVolume
    produces empty surfaces for the metric float depth used by this project.
    Outputs legacy Open3D geometry for compatibility with downstream callers.
    """

    def __init__(self, voxel_size=0.002, sdf_trunc=None):
        if voxel_size <= 0 or (sdf_trunc is not None and sdf_trunc <= 0):
            raise ValueError("TSDF voxel size and truncation must be positive")
        self.voxel_size = voxel_size
        self.sdf_trunc = 4 * voxel_size if sdf_trunc is None else sdf_trunc
        self._volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3d.core.float32,) * 3,
            attr_channels=((1,), (1,), (3,)),
            voxel_size=voxel_size, block_resolution=16, block_count=1000,
        )
        self._integrated = False

    def integrate(self, depth_img, intrinsic, extrinsic, rgb_img):
        depth = np.asarray(depth_img, dtype=np.float32)
        color = np.asarray(rgb_img, dtype=np.float32)
        if depth.shape != (intrinsic.height, intrinsic.width) or color.shape != (*depth.shape, 3):
            raise ValueError("Image dimensions must match intrinsics")
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
        if not np.any(depth):
            return
        depth_max = float(depth.max()) + 1.0
        depth = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth)))
        color = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(np.clip(color, 0, 255) / 255.0)))
        k = o3d.core.Tensor([[intrinsic.fx, 0., intrinsic.cx],
                             [0., intrinsic.fy, intrinsic.cy], [0., 0., 1.]], o3d.core.float64)
        ext = o3d.core.Tensor(extrinsic.as_matrix(), o3d.core.float64)
        multiplier = self.sdf_trunc / self.voxel_size
        blocks = self._volume.compute_unique_block_coordinates(depth, k, ext, 1.0, depth_max, multiplier)
        if blocks.shape[0] == 0:
            return
        self._volume.integrate(blocks, depth, color, k, ext, 1.0, depth_max, multiplier)
        self._integrated = True

    def get_cloud(self):
        if not self._integrated:
            return o3d.geometry.PointCloud()
        return self._volume.extract_point_cloud(weight_threshold=0.0).to_legacy()

    def get_mesh(self):
        if not self._integrated:
            return o3d.geometry.TriangleMesh()
        return self._volume.extract_triangle_mesh(weight_threshold=0.0).to_legacy()


def create_tsdf(size, resolution, depth_imgs, intrinsic, extrinsics, rgb_imgs):
    """Legacy constructor; size/resolution now specifies voxel size, not bounds."""
    tsdf = ScalableTSDFVolume(size / resolution)
    for depth_img, rgb_img, ext in zip(depth_imgs, rgb_imgs, extrinsics):
        tsdf.integrate(depth_img, intrinsic, Transform.from_list(ext), rgb_img)
    return tsdf


def _observation_frames(observation, max_distance, use_segmentation_masks):
    k = observation["intrinsics"]
    if k.fx <= 0 or k.fy <= 0:
        raise ValueError("Focal lengths must be positive")
    if max_distance is not None and (not np.isfinite(max_distance) or max_distance <= 0):
        raise ValueError("max_distance must be positive or None")
    depths, colors, extrinsics = (observation[key] for key in ("depth_imgs", "rgb_imgs", "extrinsics"))
    if not (len(depths) == len(colors) == len(extrinsics)):
        raise ValueError("Depth, color and pose counts must match")
    masks = observation["segmentation_masks"] if use_segmentation_masks else [None] * len(depths)
    if len(masks) != len(depths):
        raise ValueError("Mask and depth counts must match")
    ys, xs = np.indices((k.height, k.width))
    rays = np.stack(((xs - k.cx) / k.fx, (ys - k.cy) / k.fy, np.ones_like(xs)), axis=-1)
    for depth, color, ext, mask in zip(depths, colors, extrinsics, masks):
        depth = np.asarray(depth, dtype=np.float32)
        color = np.asarray(color)
        if depth.shape != (k.height, k.width) or color.shape != (*depth.shape, 3):
            raise ValueError("Image dimensions must match intrinsics")
        valid = np.isfinite(depth) & (depth > 0)
        if max_distance is not None:
            valid &= depth * np.linalg.norm(rays, axis=-1) <= max_distance
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != depth.shape:
                raise ValueError("Mask dimensions must match depth")
            valid &= mask
        yield np.where(valid, depth, 0).astype(np.float32), color, Transform.from_list(ext), rays, valid


def pointcloud_from_observation(observation, method="direct", max_distance=1.5,
                                voxel_size=0.002, sdf_trunc=None,
                                use_segmentation_masks=False):
    """Build a world-frame cloud without cropping, cleanup, or hole filling.

    Direct projection preserves valid samples and concatenates multiple views.
    TSDF fusion is explicit. max_distance is Euclidean camera distance in metres.
    Input colors are RGB. Empty observations return an empty PointCloud.
    voxel_size and sdf_trunc apply only to TSDF reconstruction.
    """
    if method not in ("direct", "tsdf"):
        raise ValueError("method must be 'direct' or 'tsdf'")
    volume = ScalableTSDFVolume(voxel_size, sdf_trunc) if method == "tsdf" else None
    cloud = o3d.geometry.PointCloud()
    for depth, color, ext, rays, valid in _observation_frames(observation, max_distance, use_segmentation_masks):
        if not np.any(valid):
            continue
        if volume is not None:
            volume.integrate(depth, observation["intrinsics"], ext, color)
        else:
            frame = o3d.geometry.PointCloud()
            frame.points = o3d.utility.Vector3dVector(rays[valid] * depth[valid, None])
            frame.colors = o3d.utility.Vector3dVector(np.clip(color[valid], 0, 255).astype(float) / 255)
            frame.transform(ext.inverse().as_matrix())
            cloud += frame
    return volume.get_cloud() if volume is not None else cloud


def crop_pointcloud(cloud, lower_bounds, upper_bounds):
    """Apply a world-frame workspace constraint separately from reconstruction."""
    lower, upper = np.asarray(lower_bounds), np.asarray(upper_bounds)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("Workspace bounds must be increasing 3-vectors")
    return cloud.crop(o3d.geometry.AxisAlignedBoundingBox(lower, upper))


def remove_outliers(cloud, nb_neighbors=20, std_ratio=2.0, radius=None, nb_points=12):
    """Independently optional statistical and radius filters; do not mutate input."""
    from copy import deepcopy
    cloud = deepcopy(cloud)
    if nb_neighbors is not None and len(cloud.points) > nb_neighbors:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors, std_ratio)
    if radius is not None and len(cloud.points):
        cloud, _ = cloud.remove_radius_outlier(nb_points, radius)
    return cloud


def downsample_pointcloud(cloud, voxel_size=0.002):
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    return cloud.voxel_down_sample(voxel_size)


def estimate_normals(cloud, observation, radius=0.02, max_nn=60):
    """Return a copy with normals facing the observing camera(s)."""
    from copy import deepcopy
    cloud = deepcopy(cloud)
    if len(cloud.points) < 3 or len(observation["extrinsics"]) == 0:
        return cloud
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    cloud.normalize_normals()
    if len(observation["extrinsics"]) == 1:
        eye = Transform.from_list(observation["extrinsics"][0]).inverse().translation
        cloud.orient_normals_towards_camera_location(eye)
    else:
        cloud = orient_normals_visible_simple(cloud, _build_frames(
            observation["depth_imgs"], observation["intrinsics"], observation["extrinsics"]))
    return cloud


def _build_frames(depth_imgs, intrinsic, extrinsics):
    """Pack the per-frame data we need for visibility checks."""
    K = np.array([[intrinsic.fx, 0, intrinsic.cx],
                  [0, intrinsic.fy, intrinsic.cy],
                  [0, 0, 1]], dtype=float)
    W, H = intrinsic.width, intrinsic.height
    frames = []
    for D, ext in zip(depth_imgs, extrinsics):
        ext = Transform.from_list(ext)
        frames.append({
            "K": K,
            "size": (W, H),
            "Twc": ext.inverse().as_matrix(),  # cam->world
            "Tcw": ext.as_matrix(),            # world->cam
            "depth": D.astype(np.float64),
        })
    return frames


def orient_normals_visible_simple(
    pcd: o3d.geometry.PointCloud,
    frames: List[dict],
    depth_tol_abs: float = 0.02,   # 2 cm absolute tolerance
    depth_tol_rel: float = 0.02,   # 2% of depth (helps at distance)
):
    """Fast, compact multi-view normal orientation with depth visibility checks."""
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30)
        )

    pts  = np.asarray(pcd.points)
    nors = np.asarray(pcd.normals)
    N    = pts.shape[0]
    eps  = 1e-12

    v_sum = np.zeros((N, 3), dtype=np.float64)
    w_sum = np.zeros((N,),    dtype=np.float64)
    cams  = np.stack([fr["Twc"][:3, 3] for fr in frames], axis=0)  # (M,3)

    for fr in frames:
        R = fr["Tcw"][:3, :3]; t = fr["Tcw"][:3, 3]
        Pc = pts @ R.T + t               # (N,3) world->cam
        Z  = Pc[:, 2]
        zpos = Z > 0
        if not np.any(zpos):
            continue

        fx, fy, cx, cy = fr["K"][0,0], fr["K"][1,1], fr["K"][0,2], fr["K"][1,2]
        u = (Pc[zpos, 0] * fx) / Z[zpos] + cx
        v = (Pc[zpos, 1] * fy) / Z[zpos] + cy
        W, H = fr["size"]
        inb = (u >= 0) & (v >= 0) & (u < W) & (v < H)
        if not np.any(inb):
            continue

        u  = u[inb];  v = v[inb]
        idx = np.nonzero(zpos)[0][inb]

        ui = np.floor(u).astype(np.int32)
        vi = np.floor(v).astype(np.int32)
        d_img = fr["depth"][vi, ui]
        z = Z[idx]
        tol = np.maximum(depth_tol_abs, depth_tol_rel * z)
        vis = (d_img > 0) & (np.abs(d_img - z) <= tol)
        if not np.any(vis):
            continue

        idx = idx[vis]
        c = fr["Twc"][:3, 3]          # camera center
        vdir = c - pts[idx]           # point->camera keeps outward vote
        d2 = np.einsum('ij,ij->i', vdir, vdir).clip(min=eps)
        vhat = vdir / np.sqrt(d2)[:, None]
        w = 1.0 / d2

        np.add.at(v_sum, idx, vhat * w[:, None])
        np.add.at(w_sum, idx, w)

    # Fallback: if a point had no visible votes, use all cameras (IDW)
    need_fb = (w_sum == 0)
    if np.any(need_fb):
        diff = cams[None, :, :] - pts[need_fb][:, None, :]  # (K,M,3)
        d2   = np.einsum('kmi,kmi->km', diff, diff).clip(min=eps)
        vhat = diff / np.sqrt(d2)[:, :, None]
        w    = (1.0 / d2)
        w   /= w.sum(axis=1, keepdims=True)
        v_avg_fb = (w[:, :, None] * vhat).sum(axis=1)
        v_sum[need_fb] = v_avg_fb
        w_sum[need_fb] = 1.0

    v_avg = v_sum / w_sum[:, None]
    flip = (np.einsum('ij,ij->i', nors, v_avg) < 0)
    nors[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(nors)
    return pcd


def cloud_from_image_data(
    image_data: dict,
    voxelization_size: float = 0.002,
    lower_bounds=None,
    upper_bounds=None,
    tsdf_size: float = 0.3,
    tsdf_resolution: int = 180,
    use_segmentation_masks: bool = False,
    method: str = "direct",
    max_distance: float = 1.5,
) -> Optional[o3d.geometry.PointCloud]:
    """Build, crop and clean a colored world-frame cloud with oriented normals.

    Direct projection is the default; method="tsdf" uses sparse colored fusion.
    No jitter is applied. Returns None for fewer than 50 surviving points.
    """
    pcd = pointcloud_from_observation(
        image_data, 
        method=method, 
        max_distance=max_distance,
        voxel_size=tsdf_size / tsdf_resolution,
        use_segmentation_masks=use_segmentation_masks)
    if lower_bounds is not None and upper_bounds is not None:
        pcd = crop_pointcloud(pcd, lower_bounds, upper_bounds)
    if len(pcd.points) < 50:
        return None

    pcd = remove_outliers(pcd, 
                          nb_neighbors=20, 
                          std_ratio=2.0,
                          radius=3 * voxelization_size, 
                          nb_points=12)
    if len(pcd.points) < 50:
        return None
    frames = _build_frames(image_data["depth_imgs"], image_data["intrinsics"], image_data["extrinsics"])
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=3.5 * voxelization_size, max_nn=600
        )
    )
    pcd.normalize_normals()
    pcd = orient_normals_visible_simple(pcd, frames)
    pcd = pcd.voxel_down_sample(voxel_size=voxelization_size)
    return pcd if len(pcd.points) >= 50 else None
