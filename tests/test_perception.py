"""Hardware-free regression tests for capture and reconstruction contracts."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np
import open3d as o3d

from simple_panda_grasp.utils.dataclasses import CameraIntrinsic
from simple_panda_grasp.utils.transform import Transform
from simple_panda_grasp.utils.perception_utils import (
    pointcloud_from_observation, crop_pointcloud, remove_outliers,
    downsample_pointcloud, estimate_normals, cloud_from_image_data,
)
from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.panda_constants import SCANNING_JOINTS, SCANNING_JOINTS_RANDOM, SCANNING_JOINTS_DEFAULT


def observation(depth, k=None, pose=None):
    h, w = depth.shape
    return dict(depth_imgs=np.asarray([depth], np.float32),
                rgb_imgs=np.broadcast_to(np.array([255, 0, 0], np.uint8), (1, h, w, 3)).copy(),
                intrinsics=k or CameraIntrinsic(w, h, 100, 120, w/2, h/2),
                extrinsics=np.array([(pose or Transform.identity()).to_list()]),
                eye=np.zeros((1, 3)))


class ReconstructionTests(unittest.TestCase):
    def test_projection_intrinsics_rgb_pose_and_invalid_depth(self):
        obs = observation(np.array([[1., 0., np.nan], [np.inf, -1., 1.]]),
                          CameraIntrinsic(3, 2, 2, 4, 0, 0))
        obs['extrinsics'][0, 4:] = [-2, 0, 0]
        cloud = pointcloud_from_observation(obs, max_distance=None)
        np.testing.assert_allclose(cloud.points, [[2, 0, 1], [3, .25, 1]])
        np.testing.assert_allclose(cloud.colors, [[1, 0, 0], [1, 0, 0]])

    def test_euclidean_range_and_masks(self):
        obs = observation(np.ones((1, 2)), CameraIntrinsic(2, 1, 1, 1, 0, 0))
        self.assertEqual(len(pointcloud_from_observation(obs, max_distance=1.1).points), 1)
        obs['segmentation_masks'] = np.array([[[False, True]]])
        cloud = pointcloud_from_observation(obs, max_distance=None, use_segmentation_masks=True)
        np.testing.assert_allclose(cloud.points, [[1, 0, 1]])

    def test_sparse_depth_preserved_and_multiview_concatenated(self):
        depth = np.zeros((20, 20)); depth[::2, ::2] = .5
        obs = observation(depth)
        self.assertEqual(len(pointcloud_from_observation(obs).points), 100)
        for key in ('depth_imgs', 'rgb_imgs', 'extrinsics', 'eye'):
            obs[key] = np.concatenate([obs[key], obs[key]])
        self.assertEqual(len(pointcloud_from_observation(obs).points), 200)

    def test_scalable_tsdf_outside_old_cube(self):
        obs = observation(np.full((80, 80), .503), CameraIntrinsic(80, 80, 100, 100, 40, 40))
        obs['extrinsics'][0, 4] = -1.
        cloud = pointcloud_from_observation(obs, method='tsdf', voxel_size=.01)
        self.assertGreater(len(cloud.points), 0)
        self.assertTrue(np.all(np.asarray(cloud.points)[:, 0] > .3))
        np.testing.assert_allclose(np.asarray(cloud.points)[:, 2], .503, atol=.015)
        np.testing.assert_allclose(np.asarray(cloud.colors), np.tile([1, 0, 0], (len(cloud.points), 1)), atol=1e-6)
        self.assertEqual(len(crop_pointcloud(cloud, [0, 0, 0], [.3, .3, .3]).points), 0)

    def test_empty_small_and_normals(self):
        obs = observation(np.zeros((4, 4)))
        cloud = pointcloud_from_observation(obs)
        cloud = remove_outliers(cloud, radius=.01)
        cloud = downsample_pointcloud(cloud)
        self.assertEqual(len(estimate_normals(cloud, obs).points), 0)
        self.assertIsNone(cloud_from_image_data(obs))
        obs = observation(np.full((20, 20), .5))
        cloud = estimate_normals(pointcloud_from_observation(obs), obs, radius=.1)
        self.assertTrue(np.all(np.asarray(cloud.normals)[:, 2] < 0))

    def test_mismatched_batches_rejected(self):
        obs = observation(np.ones((2, 2)))
        obs['rgb_imgs'] = []
        with self.assertRaises(ValueError):
            pointcloud_from_observation(obs)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.panda = PandaGraspInterface.__new__(PandaGraspInterface)
        self.panda.camera = SimpleNamespace(color_intrinsics=CameraIntrinsic(2, 2, 100, 200, 1, 1))
        self.panda.get_picture = lambda: (np.full((2, 2, 3), [1, 2, 3], np.uint8), np.ones((2, 2)))
        self.panda.get_cam2world = lambda: Transform.from_list([0, 0, 0, 1, 1, 2, 3])
        self.panda.observation_viewpoints = dict(default=SCANNING_JOINTS_DEFAULT[0], top_down=[1]*7, scan=SCANNING_JOINTS)
        self.panda.T_camera2ee = self.panda.T_world2panda = Transform.identity()

    def test_current_capture_no_motion_and_dictionary(self):
        self.panda.move_jj = lambda _: self.fail('Current capture must not move')
        obs = self.panda.capture_observation()
        self.assertEqual(set(obs), {'depth_imgs', 'rgb_imgs', 'intrinsics', 'extrinsics', 'eye'})
        self.assertEqual((obs['intrinsics'].fx, obs['intrinsics'].fy), (100, 200))
        np.testing.assert_array_equal(obs['rgb_imgs'][0, 0, 0], [3, 2, 1])
        np.testing.assert_allclose(obs['extrinsics'][0, 4:], [-1, -2, -3])

    @patch('simple_panda_grasp.panda_grasp_interface.sleep')
    def test_aliases_scan_and_measured_sampled_pose(self, sleep):
        self.panda.move_jj = Mock(return_value=True)
        obs = self.panda.get_observation_images_fixed_viewpoints()
        self.assertEqual(obs['depth_imgs'].shape, (3, 2, 2))
        np.testing.assert_allclose([call.args[0] for call in self.panda.move_jj.call_args_list], SCANNING_JOINTS)
        self.panda.capture_observation('default')
        np.testing.assert_allclose(self.panda.move_jj.call_args.args[0], SCANNING_JOINTS_DEFAULT[0])
        self.assertEqual(self.panda.capture_observation('multiview')['depth_imgs'].shape[0], 3)
        for name in ('get_observation_random_viewpoint', 'get_observation_top_down'):
            self.assertEqual(getattr(self.panda, name)()['intrinsics'].fy, 200)
        with patch('simple_panda_grasp.panda_grasp_interface.choice',
                   return_value=SCANNING_JOINTS_RANDOM[2]) as choose:
            obs = self.panda.get_observation_random_viewpoint()
            choose.assert_called_once_with(SCANNING_JOINTS_RANDOM)
        np.testing.assert_allclose(self.panda.move_jj.call_args.args[0], SCANNING_JOINTS_RANDOM[2])
        np.testing.assert_allclose(obs['eye'], [[1, 2, 3]])

    def test_failed_motion_stops_capture(self):
        self.panda.move_jj = lambda _: False
        self.panda.get_picture = lambda: self.fail('Must not capture after failed motion')
        with self.assertRaises(RuntimeError):
            self.panda.capture_observations()


if __name__ == '__main__':
    unittest.main()
