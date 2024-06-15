# Act*Per*MoMa

Code for the research paper: **Active-Perceptive Motion Generation for Mobile Manipulation** [1] [[Paper](https://arxiv.org/abs/2310.00433)] [[Project site](https://sites.google.com/view/actpermoma?pli=1/)]

<p float="left">
  <img src="actpermoma.gif" width="720"/>
</p>

This code is an active perception pipeline for mobile manipulators with embodied cameras to grasp in cluttered and unscructured scenes.
Specifically, it employs a receding horizon planning approach considering expected information gain and reachability of detected grasps.

This repository contains a gym-style environment for the Tiago++ mobile manipulator and uses the NVIDIA Isaac Sim simulator (Adapted from OmniIsaacGymEnvs [2]).

## Installation

__Requirements:__ The NVIDIA ISAAC Sim simulator requires a GPU with RT (RayTracing) cores. This typically means an RTX GPU. The recommended specs are provided [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html)
Besides this, in our experience, to run the Act*Per*MoMA pipeline, at least 32GB CPU RAM is needed.

### Isaac Sim

- Install isaac-sim on your PC by following the procedure outlined [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html\) \
**Note:** This code was tested on isaac-sim **version 2022.2.0**. \
[Troubleshooting](https://forums.developer.nvidia.com/t/since-2022-version-error-failed-to-create-change-watch-no-space-left-on-device/218198) (common error when starting up) 

- As we use pinocchio for kinematics [3], we need to disable isaac motion planning, because at the moment it is incompatible with pinocchio.
In your isaac installation, edit the file isaac/ov/pkg/isaac_sim-2022.2.0/apps/omni.isaac.sim.python.kit and comment out lines 541 to 548.

### Conda Environment

- Follow the isaac-sim python conda environment installation [here](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html#advanced-running-with-anaconda\) \
Note that we use a modified version of the isaac-sim conda environment `actpermoma-2022.2.0` which needs to be used instead and is available at `env_2022.2.0.yaml`. Don't forget to source the `setup_conda_env.sh` script in the isaac-sim directory before running experiments. (You could also add it to the .bashrc)

### Mushroom-RL
Install our fork of the mushroom library [4]:
```
git clone https://github.com/iROSA-lab/mushroom-rl.git
cd mushroom-rl
pip install -e .
```

### VGN
Install the devel branch of the VGN network [5]:
```
git clone -b devel https://github.com/ethz-asl/vgn.git
cd vgn
pip install -e .
```
Also download the network weights as data.zip from the VGN repo[https://github.com/ethz-asl/vgn] and put the folders models and urdfs inside the assets folder of vgn.

### robot_helpers
```
git clone https://github.com/mbreyer/robot_helpers
cd robot_helpers
pip install -e .
```
### Add the robot reachability map

Left arm: [https://hessenbox.tu-darmstadt.de/getlink/fiLmB2dHKinaEvugZrNgcuxP/smaller_full_reach_map_gripper_left_grasping_frame_torso_False_0.05.pkl]
Right arm: [https://hessenbox.tu-darmstadt.de/getlink/fiGe1B2vaHZdYZVHuovhze68/smaller_full_reach_map_gripper_right_grasping_frame_torso_False_0.05.pkl]
```
Download the reachability maps from the above links and place them in the reachability folder (<repo_root>/algos/reachability/<>)

## Launch
- Activate the conda environment:
  ```
  conda activate actpermoma-2022.2.0
  ```
- source the isaac-sim conda_setup file:
  ```
  source <PATH_TO_ISAAC_SIM>/isaac_sim-2022.2.0/setup_conda_env.sh
  ```
- run the desired method:
  ```
  python actpermoma/scripts/active_grasp_pipeline.py task=TiagoDualActivePerception train=ActPerMoMa
  ```

## References

[1]: S. Jauhri*, S. Lueth*, and G. Chalvatzaki. Active-perceptive motion generation for mobile manipulation. International Conference on Robotics and Automation (ICRA 2024), 2024. \
[2]: https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs \
[3]: https://github.com/stack-of-tasks/pinocchio \
[4]: C. D’Eramo, D. Tateo, A. Bonarini, M. Restelli, and J. Peters, “Mushroom-rl: Simplifying reinforcement learning research,” JMLR, vol. 22, pp. 131:1–131:5, 2021 \
[5]: M. Breyer, J. J. Chung, L. Ott, R. Siegwart, and J. Nieto. Volumetric Grasping Network: Real-time 6 DOF Grasp Detection in Clutter. Conference on Robot Learning (CoRL 2020), 2020. \ 

## Troubleshooting

- **"[Error] [omni.physx.plugin] PhysX error: PxRigidDynamic::setGlobalPose: pose is not valid."** This error can be **ignored** for now. Isaac-sim may have some trouble handling the set_world_pose() function for RigidPrims, but this doesn't affect the experiments.
- **"[Error] no space left on device"** https://forums.developer.nvidia.com/t/since-2022-version-error-failed-to-create-change-watch-no-space-left-on-device/218198
