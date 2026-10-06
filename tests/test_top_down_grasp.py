import unittest
from unittest.mock import Mock, patch

import numpy as np
import open3d as o3d

from examples.scene_clearing_utils import release_grasp
from examples.scene_clearing import TopDownGraspSampler, build_arg_parser
from simple_panda_grasp.panda_constants import RELEASE_JOINTS_FRONT, RELEASE_JOINTS_WIDE


class TopDownGraspTests(unittest.TestCase):
    def test_sampler_pose_width_and_top_layer(self):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector([[0, 0, .1], [.04, 0, .1], [0, .03, .1], [0, 0, 0]])
        grasps = TopDownGraspSampler(seed=0)(cloud, 5)
        self.assertEqual(len(grasps), 5)
        for grasp in grasps:
            r = grasp['pose'][:3, :3]
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)
            self.assertAlmostEqual(np.linalg.det(r), 1, places=6)
            np.testing.assert_allclose(r[:, 2], [0, 0, -1])
            self.assertAlmostEqual(grasp['pose'][2, 3], .09, places=6)
            self.assertGreaterEqual(grasp['width'], .01)
            self.assertLessEqual(grasp['width'], .08)

    def test_example_uses_complete_perception_pipeline(self):
        from examples import scene_clearing as example
        panda = Mock()
        panda.lower = np.array([0., 0., -.02])
        panda.upper = np.array([.3, .3, .28])
        panda.capture_observation.return_value = {"observation": "sentinel"}
        with patch.object(example, 'PandaGraspInterface', return_value=panda), \
             patch.object(example, 'cloud_from_image_data', return_value=None) as build_cloud, \
             patch('sys.argv', ['top_down_grasp.py', '--no-visualize']):
            example.main()
        panda.capture_observation.assert_called_once_with(viewpoint='default')
        self.assertIs(build_cloud.call_args.args[0], panda.capture_observation.return_value)
        np.testing.assert_allclose(build_cloud.call_args.kwargs['lower_bounds'], [0, 0, .01])
        panda.pick.assert_not_called()
        panda.camera.finalize.assert_called_once()

    def test_empty_cloud(self):
        self.assertEqual(TopDownGraspSampler()(o3d.geometry.PointCloud()), [])

    def test_release_targets_and_failure(self):
        self.assertEqual(build_arg_parser().parse_args([]).release_position, 'front')
        for name, joints in [('front', RELEASE_JOINTS_FRONT), ('side', RELEASE_JOINTS_WIDE)]:
            panda = Mock()
            panda.move_jj.return_value = True
            release_grasp(panda, name)
            np.testing.assert_allclose(panda.move_jj.call_args.args[0], joints)
            panda.open_gripper.assert_called_once()
        panda = Mock()
        panda.move_jj.return_value = False
        with self.assertRaises(RuntimeError):
            release_grasp(panda, 'front')
        panda.open_gripper.assert_not_called()


if __name__ == '__main__':
    unittest.main()
