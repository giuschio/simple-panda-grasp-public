"""Color preservation through each package's complete public pipeline."""
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sources' / 'grasp-env'))
from grasp_env.environments import perception_utils as sim
from simple_panda_grasp.utils import perception_utils as panda
from simple_panda_grasp.utils.dataclasses import CameraIntrinsic


def observation():
    color = np.empty((1, 80, 80, 3), np.uint8)
    color[:] = [220, 80, 30]
    return dict(depth_imgs=np.full((1, 80, 80), .503, np.float32),
                rgb_imgs=color, intrinsics=CameraIntrinsic(80, 80, 100, 120, 40, 40),
                extrinsics=np.array([[0., 0., 0., 1., -1., 0., 0.]]),
                eye=np.array([[1., 0., 0.]]))


class CloudAPIParityTests(unittest.TestCase):
    def test_original_simulation_signature_and_no_panda_noise(self):
        self.assertEqual(list(inspect.signature(sim.cloud_from_image_data).parameters),
            ['image_data', 'voxelization_size', 'lower_bounds', 'upper_bounds',
             'tsdf_size', 'tsdf_resolution', 'noise_level'])
        self.assertNotIn('noise_level', inspect.signature(panda.cloud_from_image_data).parameters)

    def test_colors_normals_and_world_frame(self):
        for function, options in ((panda.cloud_from_image_data, {}),
                                  (panda.cloud_from_image_data, {'method': 'tsdf'}),
                                  (sim.cloud_from_image_data, {'noise_level': 0.})):
            with self.subTest(function=function.__module__, options=options):
                cloud = function(observation(), lower_bounds=[.7, -.3, .4],
                    upper_bounds=[1.3, .3, .6], **options)
                self.assertIsNotNone(cloud)
                self.assertTrue(cloud.has_colors() and cloud.has_normals())
                expected = np.tile(np.array([220, 80, 30])/255., (len(cloud.points), 1))
                np.testing.assert_allclose(cloud.colors, expected, atol=1e-6)
                self.assertTrue(np.all(np.asarray(cloud.points)[:, 0] > .3))
                self.assertTrue(np.isfinite(cloud.normals).all())


if __name__ == '__main__':
    unittest.main()
