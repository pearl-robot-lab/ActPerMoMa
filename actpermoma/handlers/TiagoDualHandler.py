import torch

# check for motion planning system early on, as pinocchio has to be imported before SimApp to avoid dependency issues


from actpermoma.handlers.base.tiagohandler import TiagoBaseHandler # might be lying somewhere else
from actpermoma.robots.articulations.tiago_dual_holo import TiagoDualHolo, TiagoDualOmniHolo # might be lying somewhere else
from actpermoma.utils.visualisation_utils import Visualizer
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.stage import get_current_stage
import numpy as np
import open3d as o3d
# from omni.isaac.sensor import _sensor
# from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
from scipy.spatial.transform import Rotation
from omni.isaac.sensor import Camera
# from pxr import Usd, UsdGeom
from pxr import UsdPhysics

from omni.physx.scripts import utils, physicsUtils


# optional: comment out if you don't need it
from robot_helpers.perception import CameraIntrinsic
from robot_helpers.spatial import Rotation, Transform

# Whole Body robot handler for the dual-armed Tiago robot
class TiagoDualHandler(TiagoBaseHandler):
    def __init__(self, use_torso, sim_config, num_envs, device, usd_path=None, intrinsics=None, voxel_size=0.10, motion_planner='pinocchio'):
                                                                                                                                            # 'lula'
        
        self.motion_planner = motion_planner
        if motion_planner == 'pinocchio':
            from actpermoma.tasks.utils.pinoc_utils import PinTiagoIKSolver
            self._ik_solver_right = PinTiagoIKSolver(move_group='arm_right', include_torso=use_torso, include_base=True, include_head=True, max_rot_vel=100.0)
            self._ik_solver_left = PinTiagoIKSolver(move_group='arm_left', include_torso=use_torso, include_base=True, include_head=True, max_rot_vel=100.0)

            # no torso, no head, just base_rotation and arm joints
            self._ik_solver_right_arm_rot = PinTiagoIKSolver(move_group='arm_right', include_base_rotation=True, max_rot_vel=100.0) # no torso, no head, just base_rotation and arm joints
            self._ik_solver_left_arm_rot = PinTiagoIKSolver(move_group='arm_left', include_base_rotation=True, max_rot_vel=100.0)

            # no torso, no head, no base_rotation, just arm joints
            self._ik_solver_right_arm = PinTiagoIKSolver(move_group='arm_right', max_rot_vel=100.0)
            self._ik_solver_left_arm = PinTiagoIKSolver(move_group='arm_left', max_rot_vel=100.0)

        elif motion_planner == 'lula':
            pass #TBA

        self._use_torso = use_torso
        self._sim_config = sim_config
        self._num_envs = num_envs
        self._robot_positions = torch.tensor([0, 0, 0])  # placement of the robot in the world
        self._device = device
        self.intrinsics = intrinsics # list [fx, fy, cx, cy]
        self._cam_int = self._create_cam_intrinsics_to(intrinsics) # this is just for an application where we need this version of intrinsics, comment out if not wanted
        if usd_path is None:
            self.usd_path = "omniverse://localhost/Isaac/Robots/Tiago/tiago_dual_holobase_zed.usd"
        else:
            self.usd_path = usd_path

        # Start at 'home' positions
        self.arm_left_start = torch.tensor([0.0, 1.5708, 1.5708,
                                            1.5708, 1.5708, -1.5708, 1.5708], device=self._device)
        self.arm_right_start = torch.tensor([0.0, 1.5708, 1.5708,
                                             1.5708, 1.5708, -1.5708, 1.5708], device=self._device)
        self.gripper_left_start = torch.tensor([0.045, 0.045], device=self._device)  # Opened gripper by default
        self.gripper_right_start = torch.tensor([0.045, 0.045], device=self._device)  # Opened gripper by default
        self.gripper_left_gains = torch.tensor([1000.0, 10000.0], device=self._device) # kp and kd (Damping & Stiffness)
        self.gripper_right_gains = torch.tensor([1000.0, 10000.0], device=self._device) # kp and kd (Damping & Stiffness)
        
        self.torso_fixed_state = torch.tensor([0.25], device=self._device)

        self.default_zero_env_path = "/World/envs/env_0"

        # self.max_arm_vel = torch.tensor(self._sim_config.task_config["env"]["max_rot_vel"], device=self._device)
        # self.max_base_rot_vel = torch.tensor(self._sim_config.task_config["env"]["max_rot_vel"], device=self._device)
        # self.max_base_xy_vel = torch.tensor(self._sim_config.task_config["env"]["max_base_xy_vel"], device=self._device)
        # Get dt for integrating velocity commands
        self.dt = torch.tensor(
            self._sim_config.task_config["sim"]["dt"] * self._sim_config.task_config["env"]["controlFrequencyInv"],
            device=self._device)

        # articulation View will be created later
        self.robots = None

        # joint names
        self._base_joint_names = ["X",
                                  "Y",
                                  "R"]
        self._torso_joint_name = ["torso_lift_joint"]
        self._arm_left_names = []
        self._arm_right_names = []
        for i in range(7):
            self._arm_left_names.append(f"arm_left_{i + 1}_joint")
            self._arm_right_names.append(f"arm_right_{i + 1}_joint")

        # Future: Use end-effector link names and get their poses and velocities from Isaac
        self.ee_left_prim = ["gripper_left_grasping_frame"]
        self.ee_right_prim = ["gripper_right_grasping_frame"]
        # self.ee_left_tf =  torch.tensor([[ 0., 0.,  1.,  0.      ], # left_7_link to ee tf
        #                                  [ 0., 1.,  0.,  0.      ],
        #                                  [-1., 0.,  0., -0.196575],
        #                                  [ 0., 0.,  0.,  1.      ]])
        # self.ee_right_tf = torch.tensor([[ 0., 0., -1.,  0.      ], # right_7_link to ee tf
        #                                  [ 0., 1.,  0.,  0.      ],
        #                                  [ 1., 0.,  0.,  0.196575],
        #                                  [ 0., 0.,  0.,  1.      ]])
        
        # self._gripper_left_names = ['gripper_left_finger_joint', 'gripper_left_right_outer_knuckle_joint',
        #                             'gripper_left_left_inner_knuckle_joint',
        #                             'gripper_left_right_inner_knuckle_joint',
        #                             'gripper_left_left_inner_finger_joint', # -1  multiplier
        #                             'gripper_left_right_inner_finger_joint'] # -1  multiplier
        # self._gripper_right_names = ['gripper_right_finger_joint', 'gripper_right_right_outer_knuckle_joint',
        #                             'gripper_right_left_inner_knuckle_joint',
        #                             'gripper_right_right_inner_knuckle_joint',
        #                             'gripper_right_left_inner_finger_joint', # -1  multiplier
        #                             'gripper_right_right_inner_finger_joint'] # -1  multiplier        
        self._gripper_left_names = ["gripper_left_left_finger_joint", "gripper_left_right_finger_joint"]
        self._gripper_right_names = ["gripper_right_left_finger_joint", "gripper_right_right_finger_joint"]
        
        self._head_names = ["head_1_joint",
                            "head_2_joint"]
        # set starting position of head joints
        self.head_start = torch.tensor((0.0, 0.0), device=self._device)
        # values are set in post_reset after model is loaded
        self.base_dof_idxs = []
        self.torso_dof_idx = []
        self.arm_left_dof_idxs = []
        self.arm_right_dof_idxs = []
        self.gripper_left_dof_idxs = []
        self.gripper_right_dof_idxs = []
        self.head_dof_idxs = []
        self.upper_body_dof_idxs = []
        self.combined_dof_idxs = []

        # dof joint position limits
        self.torso_dof_lower = []
        self.torso_dof_upper = []
        self.arm_left_dof_lower = []
        self.arm_left_dof_upper = []
        self.arm_right_dof_lower = []
        self.arm_right_dof_upper = []
        self.head_dof_higher = []
        self.head_dof_lower = []

        # self._camera_path = "/TiagoDualHolo/zed_camera/zed_camera_center/ZedCamera"
        self._camera_path = "/TiagoDualHolo/zed_camera_center/ZedCamera"
        self.camera_path = self.default_zero_env_path + self._camera_path
        self.pcl_map = o3d.geometry.PointCloud()
        self._voxel_size = voxel_size

        self.vis = Visualizer()

        # set initial statatic transform between zed camera center and head_2_link
        # self._head_to_zed_tf = np.array([[ 1.1226e-06,  4.9981e-02, -9.9875e-01,  9.0029e-02], # tf to camera in isaac
        # [-4.8903e-12,  9.9875e-01,  4.9981e-02,  1.5353e-01],
        # [ 1.0000e+00, -5.6102e-08,  1.1212e-06, -5.7155e-05],
        # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
        self._head_to_zed_tf = np.array([[ 1.1226e-06, -4.9981e-02,  9.9875e-01,  9.0029e-02], # tf to camera normal (definition like a normal camera frame)
        [-5.8211e-12, -9.9875e-01, -4.9981e-02,  1.5353e-01],
        [ 1.0000e+00,  5.6103e-08, -1.1212e-06, -5.7155e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

    def _generate_camera(self):
        camera = Camera(prim_path=self.camera_path,
                        name=f"zed_camera",
                        frequency=120)

        if camera.is_valid():
            camera.initialize()
            camera.add_pointcloud_to_frame()
            camera.add_distance_to_image_plane_to_frame()
            camera.add_semantic_segmentation_to_frame()
            camera.add_instance_id_segmentation_to_frame()
            camera.add_instance_segmentation_to_frame()
            camera.set_clipping_range(0.05, 1000000.0)
            self.head_camera = camera
            if self.intrinsics is not None:
                self.set_intrinsics(self.intrinsics)
        else:
            RuntimeError("Camera Path is not valid")

    def get_robot(self):
        # make it in task and use handler as getter for path
        # tiago = TiagoDualOmniHolo(prim_path=self.default_zero_env_path + "/TiagoDualHolo", name="TiagoDualHolo",
        #                       usd_path=self.usd_path, translation=self._robot_positions)
        tiago = TiagoDualHolo(prim_path=self.default_zero_env_path + "/TiagoDualHolo", name="TiagoDualHolo",
                              usd_path=self.usd_path, translation=self._robot_positions)

        # Optional: Apply additional articulation settings
        self._sim_config.apply_articulation_settings("TiagoDualHolo", get_prim_at_path(tiago.prim_path),
                                                     self._sim_config.parse_actor_config("TiagoDualHolo"))
        
        # set material for gripper (Optional)
        self._setup_gripper_physics_material()

        # rob_prims = XFormPrimView(prim_paths_expr=tiago.prim_path + '/*', name='tiago_prims')._prims
        # for prim in rob_prims[4:]: 
        #     prim.GetAttribute('physxRigidBody:enableCCD').Set(True)
        #     prim.GetAttribute('physics:rigidBodyEnabled').Set(True)
        #     prim.GetAttribute('physxRigidBody:solveContact').Set(True)
            # prim.GetAttribute('physics:approximation').Set('convexDecomposition')

    # call it in setup_up_scene in Task
    def create_articulation_view(self):
        self.robots = ArticulationView(prim_paths_expr="/World/envs/.*/TiagoDualHolo", name="tiago_dual_holo_view")
        return self.robots

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        # add dof indexes
        self._set_dof_idxs()
        # set dof limits
        self._set_dof_limits()
        # set new default state for reset
        self._set_default_state()
        # # set default gains for gripper [Doesn't work?]
        # self._set_default_gains()
        # generate camera after loading sim
        self._generate_camera()
        # get stage
        self._stage = get_current_stage()

    def _set_dof_idxs(self):
        [self.base_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._base_joint_names]
        [self.torso_dof_idx.append(self.robots.get_dof_index(name)) for name in self._torso_joint_name]
        [self.arm_left_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._arm_left_names]
        [self.arm_right_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._arm_right_names]
        [self.gripper_left_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._gripper_left_names]
        [self.gripper_right_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._gripper_right_names]
        [self.head_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._head_names]
        self.upper_body_dof_idxs = []
        if self._use_torso:
            self.upper_body_dof_idxs += self.torso_dof_idx

        self.upper_body_dof_idxs += self.arm_left_dof_idxs + self.arm_right_dof_idxs

        # Future: Add end-effector prim paths
        self.combined_dof_idxs = self.base_dof_idxs + self.upper_body_dof_idxs

    def _set_dof_limits(self):  # dof position limits
        # (num_envs, num_dofs, 2)
        dof_limits = self.robots.get_dof_limits()
        dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        # set relevant joint position limit values
        self.torso_dof_lower = dof_limits_upper[self.torso_dof_idx]
        self.torso_dof_upper = dof_limits_upper[self.torso_dof_idx]
        self.arm_left_dof_lower = dof_limits_lower[self.arm_left_dof_idxs]
        self.arm_left_dof_upper = dof_limits_upper[self.arm_left_dof_idxs]
        self.arm_right_dof_lower = dof_limits_lower[self.arm_right_dof_idxs]
        self.arm_right_dof_upper = dof_limits_upper[self.arm_right_dof_idxs]
        # self.gripper_dof_lower = dof_limits_lower[self.gripper_idxs]
        # self.gripper_dof_upper = dof_limits_upper[self.gripper_idxs]
        self.head_lower = dof_limits_lower[self.head_dof_idxs]
        self.head_upper = dof_limits_upper[self.head_dof_idxs]

        # Holo base has no limits

    def _set_default_state(self):
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_pos[:, self.torso_dof_idx] = self.torso_fixed_state
        jt_pos[:, self.arm_left_dof_idxs] = self.arm_left_start
        jt_pos[:, self.arm_right_dof_idxs] = self.arm_right_start
        jt_pos[:, self.gripper_left_dof_idxs] = self.gripper_left_start
        jt_pos[:, self.gripper_right_dof_idxs] = self.gripper_right_start
        jt_pos[:, self.head_dof_idxs] = self.head_start

        self.robots.set_joints_default_state(positions=jt_pos)
    
    def _set_default_gains(self):        
        kps, kds = self.robots._default_kps, self.robots._default_kds

        # Update gains for gripper
        kps[:, self.gripper_left_dof_idxs] = self.gripper_left_gains[0]
        kds[:, self.gripper_left_dof_idxs] = self.gripper_left_gains[1]
        kps[:, self.gripper_right_dof_idxs] = self.gripper_right_gains[0]
        kds[:, self.gripper_right_dof_idxs] = self.gripper_right_gains[1]

        self.robots.set_gains(kps=kps, kds=kds)

    def _setup_gripper_physics_material(self):
        self._stage = get_current_stage()
        self.gripperPhysicsMaterialPath = "/World/Physics_Materials/GripperMaterial"
        
        utils.addRigidBodyMaterial(
            self._stage,
            self.gripperPhysicsMaterialPath,
            density=None,
            staticFriction=1.6,
            dynamicFriction=1.4,
            restitution=0.0
        )

        physicsUtils.add_physics_material_to_prim(
            self._stage,
            self._stage.GetPrimAtPath(self.default_zero_env_path + "/TiagoDualHolo/gripper_left_left_finger_link/collisions"),
            self.gripperPhysicsMaterialPath
            )
        physicsUtils.add_physics_material_to_prim(
            self._stage,
            self._stage.GetPrimAtPath(self.default_zero_env_path + "/TiagoDualHolo/gripper_left_right_finger_link/collisions"),
            self.gripperPhysicsMaterialPath
            )
        physicsUtils.add_physics_material_to_prim(
            self._stage,
            self._stage.GetPrimAtPath(self.default_zero_env_path + "/TiagoDualHolo/gripper_right_left_finger_link/collisions"),
            self.gripperPhysicsMaterialPath
            )
        physicsUtils.add_physics_material_to_prim(
            self._stage,
            self._stage.GetPrimAtPath(self.default_zero_env_path + "/TiagoDualHolo/gripper_right_right_finger_link/collisions"),
            self.gripperPhysicsMaterialPath
            )
        physicsUtils.add_physics_material_to_prim(
            self._stage,
            self._stage.GetPrimAtPath(self.default_zero_env_path + "/TiagoDualHolo/gripper_right_left_finger_link/visuals"),
            self.gripperPhysicsMaterialPath
            )
        physicsUtils.add_physics_material_to_prim(
            self._stage,
            self._stage.GetPrimAtPath(self.default_zero_env_path + "/TiagoDualHolo/gripper_right_right_finger_link/visuals"),
            self.gripperPhysicsMaterialPath
            )
        
    def apply_actions(self, actions):
        # Actions are velocity commands
        # The first three actions are the base velocities
        self.apply_base_actions(actions=actions[:, :3])
        self.apply_upper_body_actions(actions=actions[:, 3:])

    def apply_upper_body_actions(self, actions):
        # Apply actions as per the selected upper_body_dof_idxs (both_arms)
        # Velocity commands (rad/s) will be converted to next-position (rad) targets
        jt_pos = self.robots.get_joint_positions(joint_indices=self.upper_body_dof_idxs, clone=True)
        jt_pos += actions * self.dt  # create new position targets
        # self.robots.set_joint_position_targets(positions=jt_pos, joint_indices=self.upper_body_dof_idxs)
        # TEMP: Use direct joint positions
        self.robots.set_joint_positions(positions=jt_pos, joint_indices=self.upper_body_dof_idxs)
        if not self._use_torso:
            # Hack to avoid torso falling when it isn't controlled
            pos = self.torso_fixed_state.unsqueeze(dim=0)
            self.robots.set_joint_positions(positions=pos, joint_indices=self.torso_dof_idx)

    def apply_base_actions(self, actions):
        base_actions = actions.clone()
        # self.robots.set_joint_velocity_targets(velocities=base_actions, joint_indices=self.base_dof_idxs)
        # TEMP: Use direct joint positions
        jt_pos = self.robots.get_joint_positions(joint_indices=self.base_dof_idxs, clone=True)
        jt_pos += base_actions * self.dt  # create new position targets
        self.robots.set_joint_positions(positions=jt_pos, joint_indices=self.base_dof_idxs)

    def set_head_positions(self, head_position):
        # param torso_position: torch.tensor of shape (2,)
        self.robots.set_joint_positions(positions=head_position, joint_indices=self.head_dof_idxs)

    def set_torso_position(self, torso_position):
        # param torso_position: torch.tensor of shape (1,)
        self.robots.set_joint_positions(positions=torso_position, joint_indices=self.torso_dof_idx)

    def set_upper_body_positions(self, jnt_positions):
        # Set upper body joints to specific positions
        self.robots.set_joint_positions(positions=jnt_positions, joint_indices=self.upper_body_dof_idxs)
    
    def set_upper_body_position_targets(self, jnt_positions):
        # Set upper body joints to specific position TARGETS
        self.robots.set_joint_position_targets(positions=jnt_positions, joint_indices=self.upper_body_dof_idxs)  

    def set_gripper_right_positions(self, gripper_position):
        # param gripper_position: torch.tensor of shape (num_env, 2
        # targets
        # self.robots.set_joint_position_targets(positions=gripper_position, joint_indices=self.gripper_right_dof_idxs)
        # direct positions
        self.robots.set_joint_positions(positions=gripper_position, joint_indices=self.gripper_right_dof_idxs)

    def set_gripper_left_positions(self, gripper_position):
        # param gripper_position: torch.tensor of shape (num_env, 2
        # targets
        # self.robots.set_joint_position_targets(positions=gripper_position, joint_indices=self.gripper_left_dof_idxs)
        # direct positions
        self.robots.set_joint_positions(positions=gripper_position, joint_indices=self.gripper_left_dof_idxs)

    def set_gripper_right_position_targets(self, gripper_position):
        # param gripper_position: torch.tensor of shape (num_env, 2
        # targets
        self.robots.set_joint_position_targets(positions=gripper_position, joint_indices=self.gripper_right_dof_idxs)
        # direct positions
        # self.robots.set_joint_positions(positions=gripper_position, joint_indices=self.gripper_right_dof_idxs)

    def set_gripper_left_position_targets(self, gripper_position):
        # param gripper_position: torch.tensor of shape (num_env, 2
        # targets
        self.robots.set_joint_position_targets(positions=gripper_position, joint_indices=self.gripper_left_dof_idxs)
        # direct positions
        # self.robots.set_joint_positions(positions=gripper_position, joint_indices=self.gripper_left_dof_idxs)
    
    def set_gripper_right_efforts(self, gripper_effort):
        # efforts
        # gripper_effort = gripper_effort.repeat((1,3))
        # gripper_effort[:, -1] *= -1
        # gripper_effort[:, -2] *= -1
        self.robots.set_joint_efforts(efforts=gripper_effort, joint_indices=self.gripper_right_dof_idxs)

    def set_gripper_left_efforts(self, gripper_effort):
        # efforts
        # gripper_effort = gripper_effort.repeat((1,3))
        # gripper_effort[:, -1] *= -1
        # gripper_effort[:, -2] *= -1
        self.robots.set_joint_efforts(efforts=gripper_effort, joint_indices=self.gripper_left_dof_idxs)

    def set_base_positions(self, jnt_positions):
        # Set base joints to specific positions
        self.robots.set_joint_positions(positions=jnt_positions, joint_indices=self.base_dof_idxs)

    def get_gripper_left_positions(self):
        # return: torch.tensor of shape (num_envs, 2)
        return self.robots.get_joint_positions(joint_indices=self.gripper_left_dof_idxs, clone=True)

    def get_gripper_right_positions(self):
        # return: torch.tensor of shape (num_envs, 2)
        return self.robots.get_joint_positions(joint_indices=self.gripper_right_dof_idxs, clone=True)

    def get_robot_obs(self):
        # return positions and velocities of upper body and base joints
        combined_pos = self.robots.get_joint_positions(joint_indices=self.combined_dof_idxs, clone=True)
        # Base rotation continuous joint should be in range -pi to pi
        limits = (combined_pos[:, 2] > torch.pi)
        combined_pos[limits, 2] -= 2 * torch.pi
        limits = (combined_pos[:, 2] < -torch.pi)
        combined_pos[limits, 2] += 2 * torch.pi
        # NOTE: Velocities here will only be correct if the correct control mode is used!!
        combined_vel = self.robots.get_joint_velocities(joint_indices=self.combined_dof_idxs, clone=True)
        # Future: Add pose and velocity of end-effector from Isaac prims
        return torch.hstack((combined_pos, combined_vel))

    def get_arms_dof_pos(self):
        # (num_envs, num_dof)
        dof_pos = self.robots.get_joint_positions(clone=False)
        # left arm
        arm_left_pos = dof_pos[:, self.arm_left_dof_idxs]
        # right arm
        arm_right_pos = dof_pos[:, self.arm_right_dof_idxs]
        return arm_left_pos, arm_right_pos

    def get_arms_dof_vel(self):
        # (num_envs, num_dof)
        dof_vel = self.robots.get_joint_velocities(clone=False)
        # left arm
        arm_left_vel = dof_vel[:, self.arm_left_dof_idxs]
        # right arm
        arm_right_vel = dof_vel[:, self.arm_right_dof_idxs]
        return arm_left_vel, arm_right_vel

    def get_base_dof_values(self):
        base_pos = self.robots.get_joint_positions(joint_indices=self.base_dof_idxs, clone=False)
        base_vel = self.robots.get_joint_velocities(joint_indices=self.base_dof_idxs, clone=False)
        return base_pos, base_vel

    def get_torso_dof_values(self):
        torso_pos = self.robots.get_joint_positions(joint_indices=self.torso_dof_idx, clone=False)
        torso_vel = self.robots.get_joint_velocities(joint_indices=self.torso_dof_idx, clone=False)
        return torso_pos, torso_vel

    def get_head_dof_values(self):
        head_pos = self.robots.get_joint_positions(joint_indices=self.head_dof_idxs, clone=False)
        head_vel = self.robots.get_joint_velocities(joint_indices=self.head_dof_idxs, clone=False)
        return head_pos, head_vel

    ## vision methods
    def get_depth_map(self):
        cam = self.head_camera
        data = cam.get_current_frame()
        depth = data["distance_to_image_plane"]

        if depth is not None:
            depth = torch.tensor(depth)

        return depth

    def get_extrinsics(self):
        # returns a 4x4 homogeneous transformation matrix as torch tensor, i.e. T_base_cam (pose of camera in base frame): only the inverse transforms the points from base frame into the camera frame

        # AttributeError: module 'omni.isaac.core.utils.torch' has no attribute 'quats_to_rot_matrices'
        # ext_mat = np.linalg.inv(cam.get_world_pose())
        # workaround

        # camera_prim = self.head_camera.prim
        #https://forums.developer.nvidia.com/t/tutorial-lula-kinematics-world-pose-doesnt-update-doing-simulation/246436
        # extrinsic = torch.tensor(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), device=self._device).T
        pos_head, quat_head = self.get_frame_pose('head_2_link')
        extrinsic = TiagoDualHandler.pos_quat_to_hom(pos_head, quat_head) @ self._head_to_zed_tf

        return torch.FloatTensor(extrinsic, device=self._device)
    
    def get_extrinsincs_breyer(self):
        # returns a Transform object from Michael Breyer's robot_helpers
        ext_mat = self.get_extrinsics()

        return Transform.from_matrix(ext_mat)

    def get_intrinsics(self):
        int_mat = self.head_camera.get_intrinsics_matrix().detach()
        return int_mat

    def _create_cam_intrinsics_to(self, intrinsics):
        # intrinsics  [fx, fy, cx, cy]
        fx, fy, cx, cy = intrinsics
        # width, height = self.head_camera.get_resolution()
        width, height = round(cx * 2), round(cy * 2) # assuming square pixels, even pixel number

        cam_int = CameraIntrinsic(width, height, fx, fy, cx, cy)

        return cam_int

    def get_cam_int(self):
        # might return None if head_camera not yet created
        return self._cam_int # return intrinsics in robot_helper.perception format

    def set_intrinsics(self, int_coefficient):
        # param int_coefficient = [fx, fy, cx, cy]

        cam = self.head_camera
        self._set_cam_intrinsics(cam, int_coefficient[0], int_coefficient[1], int_coefficient[2], int_coefficient[3])
        self.intrinsics = int_coefficient
        self._cam_int = self._create_cam_intrinsics_to(int_coefficient)

    def _set_cam_intrinsics(self, cam, fx, fy, cx, cy):
        focal_length = cam.get_focal_length()
        horiz_aperture = cam.get_horizontal_aperture()
        width, height = cam.get_resolution()
        # Pixels are square so we can do
        horiz_aperture_old = cam.get_horizontal_aperture()
        width_old, height_old = cam.get_resolution()
        width, height = round(cx * 2), round(cy * 2)

        # adjust aperture relative to with/height, as pixel size won't change
        horiz_aperture = width / width_old * horiz_aperture_old
        vert_aperture = height / width * horiz_aperture
        # should be the same as fy * vert_aperture / height
        focal_length = fx * horiz_aperture / width

        cam.set_horizontal_aperture(horiz_aperture)
        cam.set_vertical_aperture(vert_aperture)
        cam.set_focal_length(focal_length)
        cam.set_resolution([width, height])

    # only differns in input arguments in case you want to set it indirectly
    def _set_cam_intrinsics_other(self, focal_length, horizontal_aperture, width, height):
        cam = self.head_camera
        cam.set_focal_length(focal_length)
        cam.set_horizontal_aperture(horizontal_aperture)
        cam.set_resolution(width, height)

    def get_pointcloud(self, merge_into_map=False):
        # return: pointcloud as torch tensor with shape (n_points, 3)
        cam = self.head_camera
        data = cam.get_current_frame()

        # pointcloud from data frame is empty!
        # https://forums.developer.nvidia.com/t/isaac-sim-camera-pointcloud-is-missing/242295
        # pcl_frame = data["pointcloud"]
        # seg_sem = data["segmantic_segmentation"]

        depth_array = data["distance_to_image_plane"]
        
        if depth_array is None:
            return None
        
        else:
            int_mat = self.get_intrinsics()
            ext_mat = self.get_extrinsics()
            pcl = self.pointcloud_from_depth(depth=depth_array,
                                               intrinsic_matrix=int_mat,
                                               extrinsic_matrix=ext_mat)
            
            if merge_into_map:
                if self.pcl_map.is_empty():
                    self.pcl_map.points = o3d.utility.Vector3dVector(pcl)
                    self.pcl_map = self.pcl_map.voxel_down_sample(voxel_size=self._voxel_size)

                else:
                    pcl_map = torch.tensor(np.asarray(self.pcl_map.points), device=self._device)
                    pcl_all = torch.vstack((pcl, pcl_map))
                    self.pcl_map.points = o3d.utility.Vector3dVector(pcl_all)                
                    self.pcl_map = self.pcl_map.voxel_down_sample(voxel_size=self._voxel_size)

                # debug visualization
                # o3d.visualization.draw_geometries([self.pcl_map])
            return pcl

    def empty_pcl_map(self):
        self.pcl_map = o3d.geometry.PointCloud()

    def get_pcl_map(self, voxel_size=0.01):
        # return: pointcloud map as tensor with shape (n_points, 3)
        return torch.tensor(np.asarray(self.pcl_map.points), device=self._device)

    @staticmethod
    def pointcloud_from_depth(depth: torch.tensor,
                              intrinsic_matrix: torch.tensor,
                              extrinsic_matrix: torch.tensor = None):
        # param: depth as torch tensor with shape (height, width)
        # param: intrinsic_matrix as torch tensor with shape (3, 3)
        # param: extrinsic_matrix as torch tensor with shape (4, 4)
        # return: pointcloud as torch tensor with shape (n_points, 3)

        depth = torch.tensor(depth.squeeze())

        # create mask for valid depth values
        mask = ~torch.logical_or(depth.isnan(), depth > 8000) # choose as max depth value
        
        # create indices according to np.indeces
        u_idxs = torch.arange(depth.shape[1], device=depth.device).repeat(depth.shape[0], 1)
        v_idxs = torch.arange(depth.shape[0], device=depth.device).reshape((-1, 1)).repeat(1, depth.shape[1])

        # get x, y, z values
        compressed_u_idxs = u_idxs[mask]
        compressed_v_idxs = v_idxs[mask]

        z = depth[mask]

        cx = intrinsic_matrix[0, 2]
        fx = intrinsic_matrix[0, 0]
        x = (compressed_u_idxs - cx) * z / fx
        
        cy = intrinsic_matrix[1, 2]
        fy = intrinsic_matrix[1, 1]
        y = ((compressed_v_idxs - cy) * z / fy)

        # combine and transform to fit isaac coordinate system (flip z as for eye space the camera is looking down the -z axis, flip y as we want +y pointing up not down)
        xyz = torch.vstack((x, -y, -z)) # in camera space

        if extrinsic_matrix is not None:
            xyz_hom = torch.vstack((xyz, torch.ones(z.shape, device=z.device)))
            xyz = torch.matmul(extrinsic_matrix, xyz_hom)[:3] # in world space

        return xyz.T

    def reset(self, indices, randomize=False):
        num_resets = len(indices)
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions.clone()
        jt_pos = jt_pos[0:num_resets]  # we need only num_resets rows
        if randomize:
            noise = torch_rand_float(-0.75, 0.75, jt_pos[:, self.upper_body_dof_idxs].shape, device=self._device)
            # Clip needed? dof_pos[:] = tensor_clamp(self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper)
            # jt_pos[:, self.upper_body_dof_idxs] = noise
            jt_pos[:, self.upper_body_dof_idxs] += noise  # Optional: Add to default instead
        self.robots.set_joint_positions(jt_pos, indices=indices)
        # Reset gripper efforts
        self.set_gripper_right_efforts(torch.zeros(num_resets,2))
        self.set_gripper_left_efforts(torch.zeros(num_resets,2))

        # reset pcl_map
        self.pcl_map = o3d.geometry.PointCloud()
    
    ## motion planning methods
    def get_ee_pose(self, move_group, joint_state=None):
        # move_group in 'arm_left', 'arm_right'
        # joint_state: (1, num_dof) tensor or ndarray; if given, will calculate ee pose for this joint state, otherwise for current joint state
        # returns pos & quat np(w, x, y, z), both are numpy arrays
        if joint_state is None:
            base_pos, _ = self.get_base_dof_values()
            base_pos = base_pos.cpu().numpy()

            torso_pos, _ = self.get_torso_dof_values()
            torso_pos = torso_pos.cpu().numpy()

            head_pos, _ = self.get_head_dof_values()
            head_pos = head_pos.cpu().numpy()

            if move_group is 'arm_left':
                arm_pos, _= self.get_arms_dof_pos() # (num_dof,) each

            elif move_group is 'arm_right':
                _, arm_pos = self.get_arms_dof_pos() # (num_dof,) each
            else:
                raise ValueError('The argument move_group has to be set to either \'arm_left\' or \'arm_right\'')

            arm_pos = arm_pos.cpu().numpy()    
            joint_state = np.hstack((base_pos, torso_pos, arm_pos, head_pos))

        if self.motion_planner is 'pinocchio': # don't forget to do .cpu().numpy() from torch tensors)
            joint_state = np.hstack([joint_state[:,:2], np.cos(joint_state[:, 2:3]), np.sin(joint_state[:, 2:3]), joint_state[:, 3:]]) # only for num_envs=1
            if move_group is 'arm_left':
                ee_pos, ee_quat = self._ik_solver_left.solve_fk_tiago(joint_state[0])
            else: # move_group is 'arm_right'
                ee_pos, ee_quat = self._ik_solver_right.solve_fk_tiago(joint_state[0])

        elif self.motion_planner is 'lula':
            raise NotImplementedError('MotionPlanning with lula is currently not enabled')

        return ee_pos, ee_quat

    def get_frame_pose(self, frame, joint_state=None):
        # joint_state: (num_dof,) tensor or ndarray; if given, will calculate frame pose for this joint state, otherwise for current joint state
        # frame: string, e.g. 'arm_left_7_link'
        # returns pos, quat np(w, x, y, z), both are numpy arrays
        move_group = 'arm_left' if 'left' in frame else 'arm_right' # might pose some problems with those 'gripper_left_right_link', but I don't think that should be applicable here

        zed = False
        if frame == 'zed':
            zed = True
            frame = 'head_2_link'
        if joint_state is None:
            base_pos, _ = self.get_base_dof_values()
            base_pos = base_pos.cpu().numpy()

            torso_pos, _ = self.get_torso_dof_values()
            torso_pos = torso_pos.cpu().numpy()

            head_pos, _ = self.get_head_dof_values()
            head_pos = head_pos.cpu().numpy()

            if move_group is 'arm_left':
                arm_pos, _= self.get_arms_dof_pos() # (num_dof,) each
            elif move_group is 'arm_right':
                _, arm_pos = self.get_arms_dof_pos() # (num_dof,) each
            else:
                raise ValueError('The argument move_group has to be set to either \'arm_left\' or \'arm_right\'')

            arm_pos = arm_pos.cpu().numpy()    
            joint_state = np.hstack((base_pos, torso_pos, arm_pos, head_pos))

        if self.motion_planner == 'pinocchio': # don't forget to do .cpu().numpy() from torch tensors)
            joint_state = np.hstack([joint_state[:,:2], np.cos(joint_state[:, 2:3]), np.sin(joint_state[:, 2:3]), joint_state[:, 3:]]) # only for num_envs=1
            if move_group is 'arm_left':
                ee_pos, ee_quat = self._ik_solver_left.solve_fk_tiago(joint_state[0], frame)
            elif move_group is 'arm_right':
                ee_pos, ee_quat = self._ik_solver_right.solve_fk_tiago(joint_state[0], frame)
            else:
                raise ValueError('The argument move_group has to be set to either \'arm_left\' or \'arm_right\'')

        elif self.motion_planner == 'lula':
            raise NotImplementedError('MotionPlanning with lula is currently not enabled')

        if zed:
            zed_pose = TiagoDualHandler.pos_quat_to_hom(ee_pos, ee_quat) @ self._head_to_zed_tf
            ee_pos, ee_quat = TiagoDualHandler.hom_to_pos_quat(zed_pose)

        return ee_pos, ee_quat

    def solve_arm_ik(self, des_pos, des_quat, move_group, joint_state=None, frame=None, n_trials=7, dt=0.1, pos_threshold=0.02, angle_threshold=5.*np.pi/180, verbose=False, debug=False):
        # move_group in 'arm_left', 'arm_right'
        # des_pos, des_quat in world frame
        # joint_state: (num_env, num_dof) tensor or ndarray; if given, will calculate ee pose for this joint state, otherwise for current joint state
        
        # transfrom des_pos, des_quat to robot frame
        R_pos, R_quat= self.get_frame_pose('R')
        R_trans = Transform(translation=R_pos, rotation=Rotation.from_quat([R_quat[1], R_quat[2], R_quat[3], R_quat[0]]))
        R_trans_inv = R_trans.inv()

        des_rot = Rotation.from_quat([des_quat[1], des_quat[2], des_quat[3], des_quat[0]])
        ee_trans = Transform(translation=des_pos, rotation=des_rot)
        ee_trans_R = R_trans_inv * ee_trans
        des_pos = ee_trans_R.translation
        des_quat = ee_trans_R.rotation.as_quat()[[3, 0, 1, 2]]
        
        if joint_state is None:
            if move_group is 'arm_left':
                arm_pos, _= self.get_arms_dof_pos() # (num_dof,) each
            elif move_group is 'arm_right':
                _, arm_pos = self.get_arms_dof_pos() # (num_dof,) each
            else:
                raise ValueError('The argument move_group has to be set to either \'arm_left\' or \'arm_right\'')

            arm_pos = arm_pos.cpu().numpy()
            joint_state = np.hstack((arm_pos[0]))
        
        if self.motion_planner == 'pinocchio': # don't forget to do .cpu().numpy() from torch tensors)
            if isinstance(des_pos, torch.Tensor):
                des_pos = des_pos.cpu().numpy()
            if isinstance(des_quat, torch.Tensor):
                des_quat = des_quat.cpu().numpy()
            if move_group is 'arm_left':
                res = self._ik_solver_left_arm.solve_ik_pos_tiago(des_pos, des_quat, joint_state, frame, n_trials, dt, pos_threshold, angle_threshold, verbose, debug) # only for num_envs=1
            if move_group is 'arm_right':
                res = self._ik_solver_right_arm.solve_ik_pos_tiago(des_pos, des_quat, joint_state, frame, n_trials, dt, pos_threshold, angle_threshold, verbose, debug) # only for num_envs=1

            if debug:
                success, best_q, q_list = res
            else:
                success, best_q = res

        elif self.motion_planner == 'lula':
            raise NotImplementedError('MotionPlanning with lula is currently not enabled')

        if debug:
            return success, best_q, q_list
        else:
            return success, best_q

    def solve_arm_rot_ik(self, des_pos, des_quat, move_group, joint_state=None, frame=None, n_trials=7, dt=0.1, pos_threshold=0.02, angle_threshold=5.*np.pi/180, verbose=False, debug=False):
        # move_group in 'arm_left', 'arm_right'
        # des_pos, des_quat in world frame
        # joint_state: (num_env, num_dof) tensor or ndarray; if given, init ik solver for this joint state, otherwise for current joint state

        # transfrom des_pos, des_quat to robot frame with world orientation
        R_pos, _ = self.get_frame_pose('R')
        R_trans = Transform.from_translation(R_pos)
        R_trans_inv = R_trans.inv()

        des_rot = Rotation.from_quat([des_quat[1], des_quat[2], des_quat[3], des_quat[0]])
        ee_trans = Transform(translation=des_pos, rotation=des_rot)
        ee_trans_R = R_trans_inv * ee_trans
        des_pos = ee_trans_R.translation
        des_quat = ee_trans_R.rotation.as_quat()[[3, 0, 1, 2]]
        
        if joint_state is None:
            base_pos, _ = self.get_base_dof_values()
            base_pos = base_pos.cpu().numpy()

            if move_group is 'arm_left':
                arm_pos, _= self.get_arms_dof_pos() # (num_dof,) each
            elif move_group is 'arm_right':
                _, arm_pos = self.get_arms_dof_pos() # (num_dof,) each
            else:
                raise ValueError('The argument move_group has to be set to either \'arm_left\' or \'arm_right\'')

            arm_pos = arm_pos.cpu().numpy()
            joint_state = np.hstack((np.cos(base_pos[:, 2]), np.sin(base_pos[:, 2]), arm_pos[0])) # only for num_envs=1
        
        if self.motion_planner == 'pinocchio': # don't forget to do .cpu().numpy() from torch tensors)
            if isinstance(des_pos, torch.Tensor):
                des_pos = des_pos.cpu().numpy()
            if isinstance(des_quat, torch.Tensor):
                des_quat = des_quat.cpu().numpy()
            if move_group is 'arm_left':
                res = self._ik_solver_left_arm_rot.solve_ik_pos_tiago(des_pos, des_quat, joint_state, frame, n_trials, dt, pos_threshold, angle_threshold, verbose, debug) 
            if move_group is 'arm_right':
                res = self._ik_solver_right_arm_rot.solve_ik_pos_tiago(des_pos, des_quat, joint_state, frame, n_trials, dt, pos_threshold, angle_threshold, verbose, debug)

            if debug:
                success, best_q, q_list = res
            else:
                success, best_q = res

        elif self.motion_planner == 'lula':
            raise NotImplementedError('MotionPlanning with lula is currently not enabled')

        if debug:
            return success, best_q, q_list
        else:
            return success, best_q

    def solve_ik(self, des_pos, des_quat, move_group, joint_state=None, frame=None, n_trials=7, dt=0.1, pos_threshold=0.02, angle_threshold=5.*np.pi/180, verbose=False, debug=False):
        # move_group in 'arm_left', 'arm_right'
        # des_pos, des_quat in world frame
        # joint_state: (num_env, num_dof) tensor or ndarray; if given, will calculate ee pose for this joint state, otherwise for current joint state
        if frame == 'zed':
            frame = 'head_2_link'
            des_head_2_pose = TiagoDualHandler.pos_quat_to_hom(des_pos, des_quat) @ np.linalg.inv(self._head_to_zed_tf)
            des_pos, des_quat = TiagoDualHandler.hom_to_pos_quat(des_head_2_pose)

        if joint_state is None:
            base_pos, _ = self.get_base_dof_values()
            base_pos = base_pos.cpu().numpy()

            torso_pos, _ = self.get_torso_dof_values()
            torso_pos = torso_pos.cpu().numpy()

            head_pos, _ = self.get_head_dof_values()
            head_pos = head_pos.cpu().numpy()

            if move_group is 'arm_left':
                arm_pos, _= self.get_arms_dof_pos() # (num_dof,) each
            elif move_group is 'arm_right':
                _, arm_pos = self.get_arms_dof_pos() # (num_dof,) each
            else:
                raise ValueError('The argument move_group has to be set to either \'arm_left\' or \'arm_right\'')

            arm_pos = arm_pos.cpu().numpy()    
            joint_state = np.hstack((base_pos, torso_pos, arm_pos, head_pos))
        
        if self.motion_planner == 'pinocchio': # don't forget to do .cpu().numpy() from torch tensors)
            if isinstance(des_pos, torch.Tensor):
                des_pos = des_pos.cpu().numpy()
            if isinstance(des_quat, torch.Tensor):
                des_quat = des_quat.cpu().numpy()

            joint_state = np.hstack([joint_state[:,:2], np.cos(joint_state[:, 2:3]), np.sin(joint_state[:, 2:3]), joint_state[:, 3:]]) # only for num_envs=1

            if move_group is 'arm_left':
                res = self._ik_solver_left.solve_ik_pos_tiago(des_pos, des_quat, joint_state[0], frame, n_trials, dt, pos_threshold, angle_threshold, verbose, debug)
            elif move_group is 'arm_right':
                res = self._ik_solver_right.solve_ik_pos_tiago(des_pos, des_quat, joint_state[0], frame, n_trials, dt, pos_threshold, angle_threshold, verbose, debug)
            else:
                raise ValueError('The argument move_group has to be set to either \'arm_left\' or \'arm_right\'')

            if debug:
                success, best_q, q_list = res
            else:
                success, best_q = res

        elif self.motion_planner == 'lula':
            raise NotImplementedError('MotionPlanning with lula is currently not enabled')

        if debug:
            return success, best_q, q_list
        else:
            return success, best_q
        
    def draw_frame(self, frame, joint_state=None, axis_length=0.3, point_size=5):
        """
        Draw the frame to the given or current joint state
        :param frame: string, e.g. 'arm_left_7_link'
        :param joint_state: (num_dof,) tensor or ndarray; if given, will calculate ee pose for this joint state, otherwise for current joint state
        :param axis_length: float, length of the axis in m
        :param point_size: float, size of the point size used to draw the axis
        """
        (pos, quat) = self.get_frame_pose(frame, joint_state)

        self.vis.draw_frame_pos_quat((pos, quat), point_size=point_size, axis_length=axis_length)

    def clear_drawings(self):
        from omni.isaac.debug_draw import _debug_draw as dd
        dd = dd.acquire_debug_draw_interface()
        dd.clear_lines()
        
    
    @staticmethod
    def pos_quat_to_hom(pos, quat):
        # pos: (3,) np array
        # quat: (4,) np array
        # returns: (4, 4) np array homogenous transformation matrix
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix() # Quaternion in scalar last format!!!
        T[:3, 3] = pos
        return T

    @staticmethod
    def hom_to_pos_quat(T):
        # T: (4, 4) np array homogenous transformation matrix
        # returns: pos: (3,) np array, quat: (4,) np array
        pos = T[:3, 3]
        quat_scipy = Rotation.from_matrix(T[:3, :3]).as_quat()
        quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]) # Quaternion in scalar last format!!!
        return pos, quat