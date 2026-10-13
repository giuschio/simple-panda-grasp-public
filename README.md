# Simple Panda Grasp

`simple-panda-grasp` is a grasping frontend for
[`panda-py`](https://github.com/JeanElsner/panda-py). It combines Franka control,
RealSense capture, eye-in-hand camera calibration, point-cloud reconstruction,
grasp reachability checks, and grasp execution behind one small Python
interface. It is the robot-control and deployment component of our
grasp-learning research code.

The package is entirely **ROS-independent**. On the robot side it inherits
`panda-py`'s direct `libfranka` integration; on the camera side it uses Intel's
RealSense Python API. In practical terms, connect the Panda to the control
computer with Ethernet, connect the camera over USB, and use the Python tools
in this repository—no ROS installation, nodes, topics, or launch files are
required.

The package assumes this eye-in-hand setup:

- a **Franka Emika Panda** seven-axis robot arm with the **Franka Hand**;
- an **Intel RealSense D435** rigidly mounted at the robot wrist; and
- a table workspace observed from one or more configured arm poses.

The camera integration provides interchangeable RGB-D capture backends. It can
use the D435's aligned raw depth directly or substantially improve its depth
maps with NVIDIA's learned
[`FoundationStereo`](https://github.com/NVlabs/fast-foundationstereo) model.

The code controls physical hardware. Review the configured poses, keep the
workspace clear, and be ready to stop the robot before running any script that
moves the arm.

## Associated paper and code

This repository is part of the code release for our CoRL 2026 paper:

> **Continual Learning for 6-DoF Grasp Synthesis via Experience and
> Demonstrations** \
> Giulio Schiavi, Andrei Cramariuc, Michael Pantic, and Roland Siegwart
>
> [Paper](https://arxiv.org/abs/2610.01301) |
> [Project page](https://giuschio.github.io/cl_grasping/)

The paper code is split into reusable repositories:

- **[Simulation environment](https://github.com/giuschio/grasp-env-public):**
  scene generation, data collection, and simulated grasp evaluation.
- **[Grasp detection model](https://github.com/giuschio/graspcl-model-public):**
  model training, grasp inference, recall, and continual adaptation.
- **Robot control and deployment (this repository):** camera calibration,
  real-world perception, reachability checks, and grasp execution.
- **[Paper benchmark](https://github.com/giuschio/graspcl-benchmark-public):**
  the top-level reproduction repository that combines the components above and
  provides the paper experiments.

For end-to-end reproduction, use the benchmark repository. This repository can
also be used independently to run grasps on the supported hardware.

If you use this package in your research, please cite:

```bibtex
@misc{schiavi2026continuallearning6dofgrasp,
  title         = {Continual Learning for 6-DoF Grasp Synthesis via Experience and Demonstrations},
  author        = {Giulio Schiavi and Andrei Cramariuc and Michael Pantic and Roland Siegwart},
  year          = {2026},
  eprint        = {2610.01301},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.01301}
}
```

## Installation

Python 3.10 or newer is required.

Clone the submodules:

```bash
git submodule update --init --recursive
```

Install the RealSense camera integration for the depth backend you want to use.

For the camera's aligned raw depth:

```bash
python -m pip install -e "submodules/realsense_utils2[camera]"
```

For learned FoundationStereo depth, install a suitable PyTorch build first,
then run:

```bash
python -m pip install -e "submodules/realsense_utils2[camera,foundation-stereo]"
```

Follow
`submodules/realsense_utils2/submodules/foundation_stereo/readme.md` to download
the FoundationStereo checkpoint. Finally, install this package:

```bash
python -m pip install -e .
```

`panda-python` is pinned to version 0.8.1. Version 1.0.0 can abort during motion
generation because of a nested native GIL release. To repair an existing
environment, run:

```bash
python -m pip install "panda-python==0.8.1"
```

The optional segmentation utilities require PyTorch and the extra dependencies:

```bash
python -m pip install -e ".[segmentation]"
```

## Robot setup

Connect the Panda directly to the control computer over Ethernet and the
RealSense D435 over USB. No ROS setup is needed. Before first use, enable the
Franka Control Interface and test the gripper:

```bash
python scripts/initialize_panda.py
```

The defaults assume the robot is at `172.16.0.2`. The initialization script
also contains default Franka Desk credentials; override its command-line
arguments if your setup differs.

## Camera calibration

The same script supports a camera on the wrist and a fixed camera beside the
robot.

### Eye in hand

This mode estimates both the wrist-camera pose and the table workspace pose.

1. Print [`configs/workspace.pdf`](configs/workspace.pdf) at **actual size on A3
   paper**. Do not scale it to fit the page.
2. Place the sheet flat on the table in front of the Panda.
3. Make sure the arm can move through its calibration poses without collisions.
4. Run:

```bash
python scripts/calibrate_camera.py
```

The script reads the AprilTag geometry from
`configs/config_workspace.yaml`, detects the target with OpenCV, and performs
hand-eye calibration. On success it writes the complete machine configuration
to `configs/config_panda.yaml`.

### Eye on base

Rigidly attach the hexagonal calibration prism to the hand. Its exact pose in
the hand does not need to be known, but it must not move during calibration.
Then run:

```bash
python scripts/calibrate_camera.py eye-on-base
```

The script moves through `WRIST_CALIBRATION_JOINTS`, using
`configs/config_calibration_bundle.yaml` to detect the prism. Arrange the fixed
camera so that preferably two adjacent tags remain visible across at least six
poses. The result is written to `T_camera2panda` in `configs/config_panda.yaml`.

The robot IP and camera backend are intentionally defined at the top of
`scripts/calibrate_camera.py`:

```python
PANDA_HOSTNAME = "172.16.0.2"
CAMERA_TYPE = "camera_fdm"
```

Set `CAMERA_TYPE` to `camera_raw` for aligned D435 depth or `camera_fdm` for
FoundationStereo depth.

### Generated configuration

Depending on the modes run, the configuration contains:

```yaml
hostname: 172.16.0.2
camera_type: camera_fdm
T_camera2ee: [qx, qy, qz, qw, tx, ty, tz]
T_world2panda: [qx, qy, qz, qw, tx, ty, tz]
T_camera2panda: [qx, qy, qz, qw, tx, ty, tz]
```

`T_camera2ee` maps the camera frame into the end-effector frame, and
`T_world2panda` maps the workspace frame into the Panda base frame.
`T_camera2panda` maps a fixed camera frame into the Panda base frame.
Translations are expressed in metres and quaternions use `xyzw` order.

### Registration-based fixed-camera calibration

To calibrate a fixed RGB-D camera directly against the Panda hand, place the
hand close to the camera and run:

```bash
python scripts/calibrate_world_camera.py
```

This alternative script registers the included hand model to one `camera_fdm`
point cloud, shows the fit and both robot frames, and then asks for a second
hand pose to validate the result. If accepted, it adds `T_camera2panda`
(camera frame to Panda base frame) to `configs/config_panda.yaml`. The prompt's front/back choice
disambiguates the hand's approximately symmetric geometry.

## Running the example

The scene-clearing example captures the workspace, constructs a point cloud,
samples top-down grasps, filters out unreachable proposals, and executes a
pick-and-place cycle:

```bash
python examples/scene_clearing.py --release-position front
```

The example displays grasp proposals before execution and prompts between
cycles. It is a basic demonstration, not a collision-complete motion planner.

## Python interface

Create the interface from the generated configuration:

```python
from simple_panda_grasp.panda_grasp_interface import PandaGraspInterface
from simple_panda_grasp.utils.perception_utils import cloud_from_image_data

panda = PandaGraspInterface("configs/config_panda.yaml")

observation = panda.capture_observation("default")
cloud = cloud_from_image_data(
    observation,
    lower_bounds=panda.lower,
    upper_bounds=panda.upper,
)

# Always release the camera when the program exits.
panda.camera.finalize()
```

### Observation viewpoints

| Argument | Behaviour |
| --- | --- |
| `None` | Capture at the current robot pose without moving. |
| `"default"` | Move to the configured default observation pose. |
| `"top_down"` | Move to the configured top-down pose. |
| `"sampled"` | Choose one pose from `SCANNING_JOINTS_RANDOM`. |
| `"multiview"` | Capture the complete configured scan. |

Use `capture_observations("scan")` for the same multiview scan, or pass a list
of viewpoint names. Default, top-down, and scan joint poses can be overridden
with an `observation_viewpoints` mapping in `config_panda.yaml`.

Each observation contains:

| Key | Contents |
| --- | --- |
| `rgb_imgs` | Batched `uint8` RGB images, shaped `N x H x W x 3`. |
| `depth_imgs` | Batched `float32` depth images in metres, shaped `N x H x W`. |
| `intrinsics` | A `CameraIntrinsic` instance. |
| `extrinsics` | World-to-camera transforms in `xyzw + xyz` format. |
| `eye` | Camera centres in world coordinates. |

### Point-cloud reconstruction

`pointcloud_from_observation()` performs direct back-projection by default. It
preserves valid samples, combines multiple views in the world frame, and does
not implicitly crop, denoise, fill holes, or estimate normals.

```python
from simple_panda_grasp.utils.perception_utils import pointcloud_from_observation

cloud = pointcloud_from_observation(observation, max_distance=1.5)
```

Pass `method="tsdf"` for sparse colored TSDF fusion. The optional `voxel_size`
defaults to 2 mm, and `sdf_trunc` defaults to four voxels. Direct projection is
recommended for a single view and for sparse FoundationStereo depth.

`cloud_from_image_data()` is the higher-level grasping pipeline. It reconstructs
the cloud, optionally crops it to workspace bounds, removes outliers, estimates
and orients normals, downsamples it, and returns `None` if fewer than 50 points
survive.

## Utility scripts

| Script | Purpose |
| --- | --- |
| `scripts/initialize_panda.py` | Enable control and test the Franka Hand. |
| `scripts/calibrate_camera.py` | Calibrate the wrist camera and workspace. |
| `scripts/calibrate_world_camera.py` | Calibrate a fixed world camera from the Panda hand. |
| `scripts/move_panda.py` | Move to the start and default observation poses. |
| `scripts/joint_recorder.py` | Interactively record Panda joint poses. |

## Tests

The unit tests do not require connected robot hardware:

```bash
python -m unittest discover -s tests -v
```
