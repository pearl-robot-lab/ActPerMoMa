import numpy as np
from copy import deepcopy as cp

from mushroom_rl.policy import Policy
from mushroom_rl.core import Agent

import vgn
from pathlib import Path
from vgn.detection import VGN, select_local_maxima

from actpermoma.utils.visualisation_utils import Visualizer
from actpermoma.utils.algo_utils import AABBox

from robot_helpers.spatial import Rotation, Transform

"""
This is the naive policy, going towards an object and trying to execute a grasp.
"""


def select_best_grasp(grasps, qualities):
    i = np.argmax(qualities)
    return grasps[i], qualities[i]


class NaivePolicy(Policy):
    def __init__(self, cfg):
        self.qual_threshold = cfg['quality_threshold']
        self.T = cfg['window_size'] # window size
        self.base_goals_radius_grasp = cfg['base_goals_radius_grasp']
        self.base_goals_radius_ig = cfg['base_goals_radius_ig']
        self.max_views = cfg['max_views']
        self.min_path_length = cfg['min_path_length']
        self.min_gain = cfg['min_gain']
        self.arm_activation_radius_base = cfg['arm_activation_radius_base']
        self.arm_activation_radius_goal = cfg['arm_activation_radius_goal']
        self.planner_type = cfg['planner_2d']

        self.bbox = None
        self.T_base_task = None
        self.T_task_base = None
        self.robot_xy = None
        self.robot_R = None
        self.goal_pos = None
  
        self.last_index = None # save the index chosen at the last step
        
        self.view_template = Rotation.from_matrix(np.array([[ 0, 0, 1], 
                                                            [-1, 0, 0], 
                                                            [ 0, -1, 0]]))
        self.views = [] # will be a list of tuples (x, y, theta)
        
        # handle theta coverage
        self.min_theta = None
        self.max_theta = None
        self.last_theta = None
        self.seen_theta_turns = 0 # handling crossovers at -pi,pi
        self.seen_theta_coverage = 0.0 # total angle covered by views until now

        self.best_grasp = None
        self.filtered_grasps = []
        self.stable_filtered_grasps = []
        self.nbv = None
        self.aborted = False

        self.base_to_zed_pose = Transform.from_matrix(np.array([[-1.08913264e-05, -4.99811739e-02,  9.98750160e-01, 2.12868273e-01],
                                                                [-1.00000000e+00,  5.44363617e-07, -1.08777139e-05, -9.63753913e-04],
                                                                [-2.34190445e-12, -9.98750160e-01, -4.99811739e-02, 1.38999854e+00],
                                                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))

        self.grasp_network = VGN(Path(vgn.__path__[0] + '/../../assets/models/vgn_conv.pth'))
        self.qual_hist = np.zeros((self.T,) + (40,) * 3, np.float32)
        self.vis = Visualizer()
        self.activated = False

    def activate(self, goal_pos, goal_aabbox, voxel_size):
        self.voxel_size = voxel_size
        self.bbox = AABBox(goal_aabbox[0], goal_aabbox[1])
        box_height = goal_aabbox[1,2] - goal_aabbox[0,2]
        self.T_base_task = Transform.from_translation(goal_pos - np.array([0.15, 0.15, 0]))
        self.T_task_base = self.T_base_task.inv()

        self.views = [] # will be a list of tuples (x, y, theta)
        self.best_grasp = None
        self.nbv = None
        self.aborted = False

        return True

    def __call__(self, state, action, *args):
        raise NotImplementedError

    def draw_action(self, state):
        """
        Sample an action in ``state`` using the policy.

        Args:
            state (dict):   'tsdf': current TSDF around the goal (vgn.perception.UniformTSDFVolume), extrinsic already in goal frame
                            'goal_pos': np.ndarray with size (3,) (x, y, z relative to the world frame) torch tensor?
                            'robot_xy': np.ndarray with size (2,) (x, y relative to the world frame)
                            'robot_R': np.ndarray with size (1,) (theta relative to the world frame)
                            'view_pose': robot_helpers.spatial.Transfrom
                            'obj_states': np.ndarray with size n*(6,) ...  -> only relevant for object collision avoidance

        Returns:
            The action sampled from the policy: A new 3D base position calculated back from the viewpoint as in
             np.ndarray(polar_angle, radius, yaw_angle (view angle o))
        """

        tsdf = state['tsdf'] # origin will be at the goal_pos
        self.goal_pos = state['goal_pos'] # in x, y, z relative to the world frame
        self.robot_xy = state['robot_xy'] # in x, y relative to the world frame
        self.robot_R = state['robot_R'] # in radians relative to the world frame
        self.goal_pos = state['goal_pos'] # in x, y, z relative to the world frame
        self.robot_xy = state['robot_xy'] # in x, y relative to the world frame
        self.robot_R = state['robot_R'] # in radians relative to the world frame
        
        if (not self.activated) or state['reset_agent']:
            self.activated = self.activate(self.goal_pos, state['goal_aabbox'], tsdf.voxel_size)  

        arms = np.array([0, 0])
        # calculate current view
        current_view = Transform(translation=state['view_pos'], rotation=self.view_template * Rotation.from_euler('YX', [state['view_theta'], state['view_phi']]))
        self.views.append(current_view)
        self.calculate_theta_coverage(state['view_theta'])
        self.integrate(tsdf)
        
        base_vel = self.goal_pos[:2] - self.robot_xy
        # if 0.8 m close to the object, execute grasp or abort
        print(np.linalg.norm(base_vel))
        if np.linalg.norm(base_vel) <= self.arm_activation_radius_goal:
            if self.best_grasp is not None:
                arms[0] = 1 # always choose left arm
                best_grasp = cp(self.best_grasp.pose)
                best_grasp.translation = best_grasp.translation + self.T_base_task.translation
                self.vis.draw_grasps_at(self.T_base_task * self.best_grasp.pose, self.best_grasp.width)
                return best_grasp, (self.seen_theta_coverage, 0), arms
            else:
                self.aborted = True
                return None, (self.seen_theta_coverage, None), arms

        # else move straight towards the object
        elif len(self.views) < self.max_views:
            # limit base_vel, so that robot doesn't get to close to the goal:
            if np.linalg.norm(base_vel) <= self.arm_activation_radius_goal + self.min_path_length - 0.1:
                base_vel = base_vel / np.linalg.norm(base_vel) * (np.linalg.norm(base_vel) - self.arm_activation_radius_goal + 0.1)
            nbv = self.get_nbv_from_base_vel(base_vel, up_or_down_next=False) # alwasy choose up
            self.vis.draw_frame_matrix(nbv.as_matrix(), axis_length=0.1, point_size=2)
            return nbv, (self.seen_theta_coverage, None), arms
        else:
            # no grasps were found after max number of views
            self.aborted = True
            return None, (self.seen_theta_coverage, None), arms

    def calculate_theta_coverage(self, theta):
        # handle infinitie rotation/overturning
        theta += self.seen_theta_turns * 2*np.pi

        # if thetas have different signs at the -pi/pi change
        if (self.last_theta is not None and         
           self.last_theta * theta < 0 and 
           abs(self.last_theta) > np.pi/2 and 
           abs(theta) > np.pi/2): 
            
            if np.mod(self.last_theta, 2*np.pi) < np.pi: # motion counterclockwise
                theta_step = abs(np.mod(self.last_theta, 2*np.pi) - 2 * np.pi - theta)
                theta = self.last_theta + theta_step
                self.seen_theta_turns +=1
            else: # motion clockwise, assuming never going round more than once (will be aborted before that)
                theta_step = 2*np.pi - abs(np.mod(self.last_theta, 2*np.pi) - 2 * np.pi - theta)
                theta = self.last_theta - theta_step
                self.seen_theta_turns -=1
            print('theta = last_theta + theta_step: ' + str(theta*180/np.pi) + ' = ' + str(self.last_theta*180/np.pi) + ' + ' + str(theta_step*180/np.pi))

        if self.min_theta is None:
            self.min_theta = theta
        elif theta < self.min_theta:
            self.seen_theta_coverage += abs(self.min_theta - theta)
            self.min_theta = theta

        if self.max_theta is None:
            self.max_theta = theta
        elif theta > self.max_theta:
            self.seen_theta_coverage += abs(self.max_theta - theta)
            self.max_theta = theta

        self.last_theta = theta

    def get_nbv_from_base_vel(self, base_vel, up_or_down_next):
        nbv_base_pos = self.robot_xy + base_vel
        nbv_theta = -(np.arctan2(nbv_base_pos[1] - self.goal_pos[1], nbv_base_pos[0] - self.goal_pos[0]) + np.pi)
        base_pose = Transform(translation=np.hstack((nbv_base_pos, np.array([0]))), rotation=Rotation.from_euler('Z', -nbv_theta))

        nbv_pos = base_pose * self.base_to_zed_pose
        nbv_pos.translation[2] = 1.08 if up_or_down_next else 1.42

        nbv_phi = -np.arctan2(np.linalg.norm(nbv_pos.translation[:2] - self.goal_pos[:2]), nbv_pos.translation[2] - self.goal_pos[2])

        nbv = Transform(translation=nbv_pos.translation, rotation=self.view_template * Rotation.from_euler('YX', [nbv_theta, nbv_phi]))
        return nbv

    def draw_noisy_action(self, state):
        raise NotImplementedError('Tried to draw a noisy action from the ActiveGraspPolicy')

    def integrate(self, tsdf):
        # src:https://github.com/ethz-asl/active_grasp/blob/devel/src/active_grasp/policy.py
        # predict grasps on the tsdf using the vgn net
        tsdf_grid = tsdf.get_grid()
        out = self.grasp_network.predict(tsdf_grid)
        colors = self.vis.interpolate_quality_colors(qualities=out.qual[out.qual >= self.qual_threshold].flatten(), qual_min=self.qual_threshold, opacity=0.4)
        x, y, z = np.mgrid[0:40, 0:40, 0:40][:, out.qual >= self.qual_threshold] * tsdf.voxel_size
        x, y, z = x + self.T_base_task.translation[0], y + self.T_base_task.translation[1], z + self.T_base_task.translation[2]
        pcl = np.concatenate((np.expand_dims(x, -1), np.expand_dims(y, -1), np.expand_dims(z, -1)), axis=-1)
        self.vis.draw_points(pcl=pcl, colors=colors, point_size=3)

        # record grasp quality
        t = (len(self.views) - 1) % self.T
        self.qual_hist[t, ...] = out.qual

        # grasp filtering
        self.filtered_grasps, self.filtered_qualitites = self.filter_grasps(out, tsdf)

        if len(self.filtered_grasps) > 0:
            self.best_grasp, quality = select_best_grasp(self.filtered_grasps, self.filtered_qualitites)
            for grasp in self.filtered_grasps:
                self.vis.draw_grasps_at(self.T_base_task * grasp.pose, grasp.width)
        
        else:
            self.best_grasp = None

    def filter_grasps(self, out, tsdf):
        grasps, qualities = select_local_maxima(tsdf.voxel_size, out, self.qual_threshold)

        filtered_grasps, filtered_qualities = [], []
        for grasp, quality in zip(grasps, qualities):
            pose = self.T_base_task * grasp.pose
            tip = pose.rotation.apply([0, 0, 0.05]) + pose.translation
            if self.bbox.is_inside(tip):
                # ignore grasps grasping from 3 cm above the table or under; come from below
                high_enough = pose.translation[2] > self.T_base_task.translation[2] + 0.03
                from_below = pose.rotation.as_matrix()[2, 2] > 0.0
                if high_enough and not from_below:
                    filtered_grasps.append(grasp)
                    filtered_qualities.append(quality)
        return filtered_grasps, filtered_qualities  


class NaiveAgent(Agent):
    """
    Agent that uses the ActiveGraspTrajPolicy to generate a trajectory of views and grasps to execute.
    """
    def __init__(self, mdp_info, agent_params_config):
        self._actor_last_loss = None
        policy = NaivePolicy(agent_params_config)

        # self.tiago_handler = agent_params_config['robot_handler']
        self.view_template_inv = Rotation.from_matrix(np.array([[0, -1, 0], 
                                                                [0, 0, -1], 
                                                                [1, 0, 0]]))

        super().__init__(mdp_info, policy)

    def fit(self, dataset):
        pass # this agent does not change based on experience

    def draw_action(self, state):
        """
        Return the action to execute in the given state. 
        Args:
            state (np.ndarray): the state where the agent is.
        Returns:
            The action to be executed.
        """
        if self.phi is not None:
            state = self.phi(state)

        if self.next_action is None:
            # check for empty depth map in first steps
            if state['tsdf'].get_scene_cloud().is_empty():
                return np.zeros((10,))
            
            tmp, tmp2, arms = self.policy.draw_action(state) # tmp2[0] is seen theta coverage

            if arms[0] == 1 or arms[1] == 1: # arm activation
                if tmp is not None:
                    pos = tmp.translation
                    rot = tmp.rotation.as_euler('ZYX')
                else:
                    pos = np.zeros(3)
                    rot = np.zeros(3)
                    tmp2 = [0]
                return np.hstack((pos, rot, *tmp2, arms))
            else:
                nbv = tmp
                if nbv is not None:
                    ## Transform nbv to 5D velocity
                    view_vel_trans = nbv.translation - state['view_pos'] # relative to base

                    # (psi), theta, phi defined as local z-y-x euler angles relative to the view template frame
                    nbv_rel_mat = (self.view_template_inv * nbv.rotation).as_matrix()
                    theta_nbv = np.arctan2(-nbv_rel_mat[2, 0], nbv_rel_mat[0, 0])
                    phi_nbv = np.arctan2(-nbv_rel_mat[1, 2], nbv_rel_mat[1, 1])

                    theta_vel = theta_nbv - state['view_theta']
                    if abs(theta_vel) > np.pi: theta_vel -= np.sign(theta_vel) * 2 * np.pi
                    phi_vel = phi_nbv - state['view_phi']

                    return np.hstack((view_vel_trans, theta_vel, phi_vel, [0], tmp2[0] ,[0], arms))
                else:
                    # no grasps found
                    return np.ones((10,))
        else:
            action = self.next_action
            self.next_action = None

            return action