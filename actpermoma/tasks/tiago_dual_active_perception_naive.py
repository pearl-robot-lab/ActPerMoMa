# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
from actpermoma.tasks.base.rl_task import RLTask
from actpermoma.handlers.TiagoDualNaiveHandler import TiagoDualHandler
from omni.isaac.core.objects.cone import VisualCone
from omni.isaac.core.prims import GeometryPrimView
from actpermoma.tasks.utils import scene_utils

from actpermoma.utils.files import get_usd_path
from actpermoma.utils.visualisation_utils import Visualizer
from actpermoma.utils.simple_planar_collision_avoidance import avoid_collision_2d

from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats, quat_diff_rad

from vgn.perception import UniformTSDFVolume
from robot_helpers.spatial import Rotation, Transform

# DEBUG
# import open3d as o3d
# import matplotlib.pyplot as plt

class TiagoDualActivePerceptionNaiveTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        
        self._max_episode_length = self._task_cfg["env"]["horizon"]
        self._randomize_robot_on_reset = self._task_cfg["env"]["randomize_robot_on_reset"]

        # Get dt for integrating velocity commands and checking limit violations
        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"]*self._sim_config.task_config["env"]["controlFrequencyInv"],device=self._device)

        # Environment object settings: (reset() randomizes the environment)
        # self._obstacle_names = ["mammut", "godishus"] # ShapeNet models in usd format
        self._obstacle_names = ["mammut"] # Consider only the table
        self._tabular_obstacle_mask = [True] # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)
        self._grasp_obj_names = ["004_sugar_box", "008_pudding_box", "010_potted_meat_can", "061_foam_brick"] # YCB models in usd format
        self._simple_grasp_obj_names = ["blue_block", "green_block", "red_block", "yellow_block"] # YCB models in usd format
        self._num_obstacles = min(self._task_cfg["env"]["num_obstacles"],len(self._obstacle_names))
        self._num_grasp_objs = min(self._task_cfg["env"]["num_grasp_objects"],len(self._grasp_obj_names))
        self._obj_states = torch.zeros((6*(self._num_obstacles+self._num_grasp_objs-1),self._num_envs),device=self._device) # All grasp objs except the target object will be used in obj state (BBox)
        self._obstacles = []
        self._obstacles_dimensions = []
        self._grasp_objs = []
        self._grasp_objs_dimensions = []

        # Choose num_obs and num_actions based on task
        # 3D goal/target object + 3D robot + 6D bbox for each obstacle in the room.
        # (3 pos goal + 3 pos robot + 2 (torso, head) + 6*(n-1)= 7 + )
        self._num_observations = 8 + len(self._obj_states) # updated farther down to add the pixels of the depthmap
        self._move_group = self._task_cfg["env"]["move_group"]
        self._use_torso = self._task_cfg["env"]["use_torso"]

        # Position control. Actions are base SE2 pose + head elevation angle (4) and discrete arm activation (2)
        self._num_actions = self._task_cfg["env"]["continous_actions"] + self._task_cfg["env"]["discrete_actions"]

        # env specific limits
        self._world_xy_radius = self._task_cfg["env"]["world_xy_radius"]
        self._action_xy_radius = self._task_cfg["env"]["action_xy_radius"]
        self._action_ang_lim = self._task_cfg["env"]["action_ang_lim"]
        self._action_head_2_center = self._task_cfg["env"]["action_head_2_center"]
        self._action_head_2_rad = self._task_cfg["env"]["action_head_2_rad"]

        # stores a tsdf around the goal object
        self._complete_state_tsdf = UniformTSDFVolume(0.3, 40)

        # End-effector reaching settings
        self._goal_pos_threshold = self._task_cfg["env"]["goal_pos_thresh"]
        self._goal_ang_threshold = self._task_cfg["env"]["goal_ang_thresh"]
        # For now, setting dummy goal:
        self._goals = torch.hstack((torch.tensor([[0.8,0.0,0.4+0.15]],device=self._device),euler_angles_to_quats(torch.tensor([[0.19635, 1.375, 0.19635]]),device=self._device)))[0].repeat(self.num_envs,1)
        self._goal_tf = torch.zeros((4,4),device=self._device)
        self._goal_tf[:3,:3] = torch.tensor(Rotation.from_quat(np.array([self._goals[0,3+1],self._goals[0,3+2],self._goals[0,3+3],self._goals[0,3]])).as_matrix(),dtype=float,device=self._device) # Quaternion in scalar last format!!!
        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0], self._goals[0,1], self._goals[0,2], 1.0],device=self._device) # x,y,z,1

        # robot state
        self._robot_tf = torch.eye(4, device=self._device).repeat(self._num_envs,1,1)
        self._view_pos = torch.zeros((self._num_envs,3),device=self._device)
        self._torso_state = torch.zeros(self._num_envs, device=self._device)
        self._head_2_state = torch.zeros(self._num_envs, device=self._device)
        self._theta = torch.zeros(self._num_envs, device=self._device)
        self._phi = torch.zeros(self._num_envs, device=self._device)
        self._curr_goal_tf = self._goal_tf.clone()

        self._tsdf_orig_tf = torch.eye(4, device=self._device)
        self._tsdf_orig_tf[:3,-1] = self._goal_tf[:3,-1] - 0.15
        self._tsdf_orig_tf_inv = torch.linalg.inv(self._tsdf_orig_tf)

        # Metrics
        self._is_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._is_done = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._has_just_reset = 0
        self._distance_walked = torch.zeros(self._num_envs, device=self._device)
        self._grasp_failure = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._aborted = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._num_views = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # Handler for Tiago
        cx = self._task_cfg["env"]["cx"]
        cy = self._task_cfg["env"]["cy"]
        usd_path = (get_usd_path()/ 'tiago_dual_holobase_zed/tiago_dual_holobase_zed.usd').as_posix()
        self.tiago_handler = TiagoDualHandler(move_group=self._move_group, use_torso=self._use_torso, sim_config=self._sim_config, num_envs=self._num_envs, device=self._device,
                                                usd_path=usd_path, intrinsics=[self._task_cfg["env"]["fx"], self._task_cfg["env"]["fy"], cx, cy])

        # update observation space to add #pixels
        self._depth_map_shape = (int(2 * cy), int(2 * cx))
        self._depth_map_size = int(4 * cx * cy)
        self._num_observations += self._depth_map_size

        self.vis = Visualizer()
        self.view_template = Rotation.from_matrix(np.array([[ 0,  0, 1], 
                                                            [-1,  0, 0], 
                                                            [ 0, -1, 0]]))

        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        import omni
        
        self.tiago_handler.get_robot()
        
        # Spawn obstacles (from ShapeNet usd models):
        for i in range(self._num_obstacles):
            obst = scene_utils.spawn_obstacle(self, name=self._obstacle_names[i], prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
            self._obstacles.append(obst) # Add to list of obstacles (Geometry Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor", sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=obst.prim_path)
        
        # Spawn grasp objs (from YCB usd models):
        for i in range(self._num_grasp_objs):
            grasp_obj = scene_utils.spawn_grasp_object(self, name=self._grasp_obj_names[i], prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
            self._grasp_objs.append(grasp_obj) # Add to list of grasp objects (Rigid Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor", sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=grasp_obj.prim_path)
        
        # Goal visualizer
        goal_viz = VisualCone(prim_path=self.tiago_handler.default_zero_env_path+"/goal",
                                radius=0.05,height=0.05,color=np.array([1.0,0.0,0.0]))
        super().set_up_scene(scene)
        self._robots = self.tiago_handler.create_articulation_view()
        scene.add(self._robots)
        self._goal_vizs = GeometryPrimView(prim_paths_expr="/World/envs/.*/goal",name="goal_viz")
        scene.add(self._goal_vizs)

        # Enable object axis-aligned bounding box computations
        scene.enable_bounding_boxes_computations()
        
        # Add spawned objects to scene registry and store their bounding boxes:
        for obst in self._obstacles:
            scene.add(obst)
            self._obstacles_dimensions.append(scene.compute_object_AABB(obst.name)) # Axis aligned bounding box used as dimensions
        for grasp_obj in self._grasp_objs:
            scene.add(grasp_obj)
            self._grasp_objs_dimensions.append(scene.compute_object_AABB(grasp_obj.name)) # Axis aligned bounding box used as dimensions
        
        # Optional viewport for rendering in a separate viewer
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        self.viewport_api_window = get_viewport_from_window_name("Viewport")
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport_api=self.viewport_api_window)

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        self.tiago_handler.post_reset()

    def get_observations(self):
        # Handle any pending resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # Get robot observations
        d_tmp = self.tiago_handler.get_depth_map()
        if d_tmp is not None and not d_tmp.nelement() == 0:
            curr_depth_map = d_tmp.clone().detach().flatten().unsqueeze(dim=0)
        else:
            curr_depth_map = torch.zeros((1, self._depth_map_size), dtype=torch.float, device=self._device).flatten().unsqueeze(dim=0)

        # Fill observation buffer
        # Goal: 3D pos 
        curr_goal_pos = self._goal_tf[0:3,3].unsqueeze(dim=0)
        # oriented bounding boxes of objects
        curr_bboxes_flattened = self._obj_bboxes.flatten().unsqueeze(dim=0)
        self.obs_buf = torch.hstack((curr_goal_pos, self._view_pos, self._theta, self._phi, curr_bboxes_flattened, curr_depth_map))

        return self.obs_buf

    def obs_to_state(self, observation):
        depth_map = observation[-self._depth_map_size:].reshape(self._depth_map_shape)

        state = dict()
        
        # transform extrinsics into goal frame
        extrinsic = torch.matmul(torch.linalg.inv(self.tiago_handler.get_extrinsics()), self._tsdf_orig_tf) # cam to tsdf

        # use camera intrinsics obejct of vgn
        depth_map[np.isinf(depth_map)] = 0
        depth_map = depth_map.astype(np.float32)
        self._complete_state_tsdf.integrate(depth_img=depth_map, intrinsic=self.tiago_handler.get_cam_int(),
                                                            extrinsic=Transform.from_matrix(extrinsic)) # this integrate func want the extrinsic matrix that multiplies world coordinates to camera coordinates

        self.vis.clear_all()

        # VIS: show TSDF voxels in isaac
        pcl = np.asarray(self._complete_state_tsdf.get_scene_cloud().points) + self._tsdf_orig_tf[:3, 3].numpy()
        voxel_size = self._complete_state_tsdf.voxel_size
        # self.vis.draw_voxels_from_center(pcl, voxel_size)

        # self.vis.draw_box_min_max(self._tsdf_orig_tf[:3, 3].numpy(), self._tsdf_orig_tf[:3, 3].numpy() + 0.3)

        ## DEBUG: preparation for drawing the scene cloud
        # pcl = self._complete_state_tsdf.get_scene_cloud()
        # if pcl.is_empty():
        #     pcl

        # pcl.transform(self._tsdf_orig_tf)
        # pcl.paint_uniform_color([1, 0, 0])

        # ## DEBUG
        # intrinsic=self.tiago_handler.get_cam_int()
        # o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=intrinsic.width,
        #     height=intrinsic.height,
        #     fx=intrinsic.fx,
        #     fy=intrinsic.fy,
        #     cx=intrinsic.cx,
        #     cy=intrinsic.cy)
        # extrinsic = torch.linalg.inv(self.tiago_handler.get_extrinsics())
        # d_image = o3d.geometry.Image(depth_map)
        # pcl_all = o3d.geometry.PointCloud.create_from_depth_image(d_image, o3d_intrinsic, extrinsic)
        # o3d.visualization.draw_geometries([pcl, pcl_all])

        # breakpoint()

        state['tsdf'] = self._complete_state_tsdf
        state['goal_pos'] = observation[:3]
        state['view_pos'] = observation[3:6]
        state['view_theta'] = observation[6]
        state['view_phi'] = observation[7]
        state['obj_states'] = observation[8:-self._depth_map_size]
        state['reset_agent'] = self._has_just_reset

        self._num_views[0] += 1 # Change, if last step is not a view step

        if self._has_just_reset: self._has_just_reset = 0
        return state

    def get_render(self):
        # Get ground truth viewport rgb image
        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        return np.array(gt["rgb"][:, :, :3])
    
    def pre_physics_step(self, actions) -> None:
        # actions (num_envs, num_action)
        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        arms = actions[:, 7:].numpy().flatten()

        if arms[0] == 1 or arms[1] == 1: # arm activation
            # check for abortion (no good grasps found after max steps with low information gain)

            # no real grasp would ever be at [0,0,0]
            ee_pos = actions[:, :3].cpu().numpy().squeeze()
            if (ee_pos == [0,0,0]).all():
                print('aborted')
                self._aborted[0] = 1
                self._is_done[0] = 1
                return

            ee_rot_panda = Rotation.from_euler('ZYX', actions[:, 3:6].numpy().squeeze()).as_matrix()
            ee_rot = Rotation.from_matrix(np.vstack((ee_rot_panda[:, 2], ee_rot_panda[:, 1], -ee_rot_panda[:, 0])).T) # transform to tiago grasping frame; could also be done via multiplication
            ee_quat = ee_rot.as_quat()[[3, 0, 1, 2]]
            ee_pos = ee_pos + ee_rot.apply(np.array([-0.03, 0, 0])) # adjust to tiago gripper (the links are 7 cm longer than the panda ones (12 vs 5 cm)) 

            gripper_state = actions[:, 6].numpy().squeeze()

            ee_grasp_helper = ee_pos + ee_rot.apply(np.array([-0.1, 0, 0]))

            if(arms[0] > arms[1]): # This is the arm decision variable
                ## PRE GRASP POSE
                success, best_q, q_list = self.tiago_handler.solve_ik(des_pos=ee_grasp_helper, des_quat=ee_quat,
                                        pos_threshold=0.01)

                new_X, new_Y, new_torso = best_q[[True, True, False, False, True, 
                                        False, False, False, False, 
                                        False, False, False, False, False]]
                new_R = torch.Tensor([np.arctan2(best_q[3], best_q[2])])
                new_base_xy = torch.Tensor([new_X, new_Y])
                new_arm = best_q[[False, False, False, False, False,
                                        True, True, True, True,
                                        True, True, True, False, False]]

                self._distance_walked[0] += torch.tensor(np.sqrt(np.sum(np.square(np.array([new_X, new_Y]) - self._robot_xy))))

                self._torso_state = torch.Tensor([new_torso])
                self._theta = -new_R.unsqueeze(dim=0)

                self.tiago_handler.set_base_positions(torch.hstack((new_base_xy, new_R)))
                self.tiago_handler.set_upper_body_positions(torch.hstack((self._torso_state, torch.Tensor(new_arm))))

                for _ in range(4): self._env._world.step(render=True)
                import time
                time.sleep(1)

                ## GRASP POSE: approach 
                success, best_q_new, q_list = self.tiago_handler.solve_arm_ik(des_pos=ee_pos, des_quat=ee_quat,
                                        pos_threshold=0.01)
                
                new_torso = best_q_new[0]
                new_arm = best_q_new[1:]

                self._torso_state = torch.Tensor([new_torso])
                self.tiago_handler.set_upper_body_positions(torch.hstack((self._torso_state, torch.Tensor(new_arm))))

                for _ in range(4): self._env._world.step(render=True)
                time.sleep(1)

                ## CLOSE GRIPPER
                # Close until you can't close anymore
                self.tiago_handler.set_gripper_right_efforts(torch.Tensor([-35.0,-35.0]).unsqueeze(dim=0))

                for _ in range(350): self._env._world.step(render=False)

                ## lift object 10 cm
                for q_conf in np.linspace(best_q_new, best_q[4:12], 10):
                    new_torso = q_conf[0]
                    new_arm = q_conf[1:]

                    self._torso_state = torch.Tensor([new_torso])
                    self.tiago_handler.set_upper_body_position_targets(torch.hstack((self._torso_state, torch.Tensor(new_arm))))

                    for _ in range(4): self._env._world.step(render=True)

                self._is_done[0] = 1 # in order to not continue

        else:   # for velocity control of base, torso and head_2
            view_vel_trans, theta_vel, phi_vel = actions[:, :3].numpy().flatten(), actions[:, 3].numpy(), actions[:, 4].numpy()
            # respect velocity limits
            radius = np.sqrt(np.sum(np.square(view_vel_trans[:2])))
            ratio_xy = radius / self._action_xy_radius
            ratio_R = abs(theta_vel) / self._action_ang_lim # theta = -R

            if ratio_xy > 1 or ratio_R > 1:
                ratio = max(ratio_xy, ratio_R)
                view_vel_trans, theta_vel, phi_vel = view_vel_trans/ratio, theta_vel/ratio, phi_vel/ratio
            
            # calculate desired view pose
            nbv_pos = self._view_pos[:].squeeze().numpy() + view_vel_trans
            nbv_rot = self.view_template * Rotation.from_euler('YX', [self._theta.item() + theta_vel.item(), self._phi.item() + phi_vel.item()])
            des_quat_scipy = nbv_rot.as_quat() # Quaternion in scalar last format!!!
            des_quat = np.array([des_quat_scipy[3], des_quat_scipy[0], des_quat_scipy[1], des_quat_scipy[2]])

            # get good initial IK guess
            X, Y = nbv_pos[:2]
            R = - (self._theta.item() + theta_vel.item())
            torso = self._torso_state.item() + view_vel_trans[2]
            head_2 = self._head_2_state.item() + phi_vel.item()

            # compute IK
            joint_state = np.hstack(([X, Y, R, torso], np.zeros((7,)), [0, head_2])).reshape((1, -1))
            success, best_q, q_list = self.tiago_handler.solve_ik(des_pos=nbv_pos, des_quat=des_quat, joint_state=joint_state, frame='zed')

            new_X, new_Y, new_torso, new_head_2 = best_q[[True, True, False, False, True, 
                                            False, False, False, False, 
                                            False, False, False, False, True]]
            new_X, new_Y = avoid_collision_2d(current=self._robot_xy, 
                                              radius=0.5, 
                                              step=np.array([new_X, new_Y])-self._robot_xy, 
                                              obstacle_bbox=self._obst_bbox)

            new_base_xy = torch.Tensor([new_X, new_Y])

            self._distance_walked[0] += torch.tensor(np.sqrt(np.sum(np.square(np.array([new_X, new_Y]) - self._robot_xy))))

            # ORIENT TIAGO AT GOAL
            new_R = torch.Tensor([torch.arctan2(self._goal_tf[1, 3] - new_Y, self._goal_tf[0, 3] - new_X)])

            if not success:
                print('IK failed, using best guess')

            # self._robot_tf
            self._robot_xy = np.array([new_X, new_Y])
            self._torso_state = torch.Tensor([new_torso])
            self._theta = -new_R.unsqueeze(dim=0)
            self._head_2_state = torch.Tensor([new_head_2])
            self._phi = self._head_2_state.unsqueeze(dim=0) - 0.049967

            self.tiago_handler.set_base_positions(torch.hstack((new_base_xy, new_R)))
            self.tiago_handler.set_torso_position(self._torso_state)
            self.tiago_handler.set_head_positions(torch.hstack((torch.zeros(new_head_2.shape), self._head_2_state)))

            cam_pos, _ = self.tiago_handler.get_frame_pose('zed')
            self._view_pos = torch.tensor(cam_pos, device=self._device).unsqueeze(dim=0)

            # Transform goal to robot frame
            inv_base_tf = torch.linalg.inv(self._robot_tf)
            self._curr_goal_tf = torch.matmul(inv_base_tf,self._goal_tf)      

    def reset_idx(self, env_ids):
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # reset dof values
        self.tiago_handler.reset(indices,randomize=self._randomize_robot_on_reset)
        self._torso_state = self.tiago_handler.get_torso_dof_values()[0].flatten()
        # reset the scene objects (randomize), get target end-effector goal/grasp as well as oriented bounding boxes of all other objects
        self._curr_grasp_obj, self._goals[env_ids], self._obj_bboxes, self._obst_bbox = scene_utils.setup_tabular_scene(
                                self, self._obstacles, self._tabular_obstacle_mask[0:self._num_obstacles], self._grasp_objs,
                                self._obstacles_dimensions, self._grasp_objs_dimensions, self._world_xy_radius, self._device, return_obstacles=True)
        self._curr_obj_bboxes = self._obj_bboxes.clone()

        self._goal_tf = torch.zeros((4,4),device=self._device)
        goal_rot = Rotation.from_quat(np.array([self._goals[0,3+1],self._goals[0,3+2],self._goals[0,3+3],self._goals[0,3]])) # Quaternion in scalar last format!!!
        self._goal_tf[:3,:3] = torch.tensor(goal_rot.as_matrix(),dtype=float,device=self._device)
        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0], self._goals[0,1], self._goals[0,2] + 0.01, 1.0],device=self._device) # x,y,z,1
        self._curr_goal_tf = self._goal_tf.clone().unsqueeze(dim=0)
        self._goals_xy_dist = torch.linalg.norm(self._goals[:,0:2],dim=1) # distance from origin
        # Pitch visualizer by 90 degrees for aesthetics
        goal_viz_rot = goal_rot * Rotation.from_euler("xyz", [0,np.pi/2.0,0])
        self._goal_vizs.set_world_poses(indices=indices,positions=self._goals[env_ids,:3] + torch.tensor([[0, 0, 0.4]], device=self._device),
                orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]],device=self._device).unsqueeze(dim=0))

        # orient tiago towards goal
        R = torch.arctan2(self._curr_goal_tf[:, 1, 3], self._curr_goal_tf[:, 0, 3])
        self.tiago_handler.set_base_positions(torch.hstack((torch.tensor([[0, 0]]), 
                    R.unsqueeze(dim=0))))
        self._robot_tf[:, :3, :3] = torch.tensor(Rotation.from_euler("Z", R).as_matrix(), device=self._device).unsqueeze(dim=0)
        self._theta = -R.unsqueeze(dim=0)
        self._robot_xy = np.array([0, 0])

        # have tiago look at goal
        cam_pos, _ = self.tiago_handler.get_frame_pose('zed')
        z_diff = cam_pos[2] - self._goal_tf[2, 3]
        xy_dist = np.sqrt((cam_pos[0] - self._goal_tf[0, 3])**2 + (cam_pos[1] - self._goal_tf[1, 3])**2)
        alpha = np.arctan2(z_diff, xy_dist)

        # set state
        self._head_2_state = torch.tensor([-alpha],device=self._device)
        self._phi = self._head_2_state.unsqueeze(dim=0) - 0.049967
        head_2 = self._head_2_state.unsqueeze(dim=0)
        self.tiago_handler.set_head_positions(torch.hstack((torch.zeros(head_2.shape), head_2)))
        self._view_pos = torch.tensor(cam_pos, device=self._device).unsqueeze(dim=0)
        
        # set the tsdf origin to start 15 cm (in each x, y) away from the goal, orientation is same as world/base
        self._tsdf_orig_tf[:2,-1] = self._goal_tf[:2,-1] - 0.15
        self._tsdf_orig_tf[2,-1] = self._goal_tf[2,-1]
        self._tsdf_orig_tf_inv = torch.linalg.inv(self._tsdf_orig_tf)
        
        # bookkeeping
        self._is_success[env_ids] = 0
        self._is_done[env_ids] = 0
        self._distance_walked[env_ids] = 0
        self._grasp_failure[env_ids] = 0
        self._aborted[env_ids] = 0
        self._num_views[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.extras[env_ids] = {}
        self._has_just_reset = 1

        self._complete_state_tsdf = UniformTSDFVolume(0.3, 40)

        for _ in range(5): self._env._world.step(render=True) # Step the world to update the scene
    
    def calculate_metrics(self) -> None:
        self.rew_buf[:] = self._is_success.clone()
        self.extras = {'distance': self._distance_walked.clone(), # in total, i.e. including the one to the grasp (lay line) 
                          'grasp_success': self._is_success.clone(), 
                          'grasp_failure': self._grasp_failure.clone(), 
                          'aborted': self._aborted.clone(),
                          'num_views': self._num_views.clone()}

    def is_done(self) -> None:
        resets = self._is_done.clone()
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets

    # only for debugging
    def interact(self):
        import time
        try:
            while True: 
                self._env._world.step(render=True)
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
    
    # only for debugging
    def interact_render(self):
        import time
        try:
            while True: 
                self._env._world.render()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
