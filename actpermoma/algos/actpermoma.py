import numpy as np
import itertools
import torch
from copy import deepcopy as cp
import signal
from pathlib import Path

from mushroom_rl.policy import Policy
from mushroom_rl.core import Agent

import vgn
from vgn.detection import VGN, select_local_maxima

from robot_helpers.spatial import Rotation, Transform

from actpermoma.utils.visualisation_utils import Visualizer
from actpermoma.utils.algo_utils import ViewHalfSphere, AABBox



from .reachability.reachability_evaluator import ReachabilityEvaluator


"""
This is the ActPerMoMa policy.
"""

def raycast_on_voxel_array_torch(
    voxel_index_array,
    voxel_size,
    tsdf_grid,
    ori,
    pos,
    fx,
    fy,
    cx,
    cy,
    u_min,
    u_max,
    u_step,
    v_min,
    v_max,
    v_step,
    t_min,
    t_max,
    t_step,
    device='cpu'
):
    # voxel_index_array is a large precomputed array. We fill in the visited voxels
    pixel_grid = torch.meshgrid(torch.arange(u_min, u_max, u_step, device=device), torch.arange(v_min, v_max, v_step, device=device))
    pixels = torch.dstack((pixel_grid[0],pixel_grid[1])).reshape(-1, 2)
    directions = torch.vstack(((pixels[:,0] - cx) / fx, (pixels[:,1] - cy) / fy, torch.ones((1,pixels.shape[0]),device=device))).T # Homogenous co-ordinates
    directions_normed = directions / torch.norm(directions, dim=1).unsqueeze(dim=1)
    # Start from given origin
    directions = torch.matmul(torch.tensor(ori,device=device,dtype=torch.float32), directions_normed.T).T

    t_range = torch.arange(t_min, t_max, t_step, device=device)
    # calc query positions for all rays
    p_proposal = torch.tensor(pos, device=device) + t_range[None,:,None] * directions[:,None,:] # rays x steps x 3
    proposal_indices = (p_proposal / torch.tensor(voxel_size, device=device)).to(torch.int64)
    indices_mask = torch.all((proposal_indices > -1) & (proposal_indices < 40), axis=2)

    tsdf_vals = -torch.ones((p_proposal.shape[0],p_proposal.shape[1]), device=device)
    tsdf_grid_t = torch.tensor(tsdf_grid, device=device)
    i, j, k = proposal_indices[indices_mask].T
    tsdf_vals[indices_mask] = tsdf_grid_t[i, j, k]
    ## get elements up-to surface crossings
    # get sign matrix (add ones at the end for keeping the same shape)
    sign_matrix = torch.cat([torch.sign(tsdf_vals[:,:-1] * tsdf_vals[:,1:]), torch.ones((tsdf_vals.shape[0], 1),device=device)], dim=-1)
    cost_matrix = sign_matrix * torch.arange(tsdf_vals.shape[1], 0, -1, device=device)
    # get first crossing
    crossing_indices = torch.argmin(cost_matrix, dim=1)
    # get all elements up to (and including) first crossing
    mask_per_ray = torch.arange(tsdf_vals.shape[1], device=device) <= crossing_indices.unsqueeze(dim=1)
    # get elements with tsdf values above -1.0 and below 0.0
    mask_vals = (tsdf_vals > -1.0) & (tsdf_vals < 0.0)
    # fianlly, get elements that count
    voxel_index_count = torch.sum(mask_per_ray & mask_vals).cpu().numpy()
    voxel_index_array[:voxel_index_count] = proposal_indices[mask_per_ray & mask_vals].cpu().numpy()

    # return voxel_indices
    return voxel_index_count, voxel_index_array

# @jit(nopython=True)
def ig_fn_parallel_jit(tsdf_grid, voxel_size, fx, fy, cx, cy, u_step, v_step, bbox_corners, self_bbox_min, self_bbox_max, T_task_base_np, view_nps, debug=False, vis=None, T=None):
    igs = []
    for view in view_nps:
        # Project bbox onto the image plane to get better bounds
        T_cam_base = np.linalg.inv(view)
        corners = ((T_cam_base[:3,:3] @ bbox_corners.T) + (T_cam_base[:3,3:4]))        
        u = (fx * corners[0] / corners[2] + cx)
        np.round(u,0,u)
        u = u.astype(np.int64)
        v = (fy * corners[1] / corners[2] + cy)
        np.round(v,0,v)
        v = v.astype(np.int64)
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()

        t_min = corners[2].min() - 0.3 # A bit less than the min corner # 0.0
        t_max = corners[2].max() + 0.2  # This bound might be a bit too short
        t_step = np.sqrt(3) * voxel_size  # Could be replaced with line rasterization

        # Cast rays from the camera view (we'll work in the task frame from now on)
        view = T_task_base_np @ view
        ori, pos = view[:3,:3], view[:3,3]
        
        ## Count rear side voxels within the bounding box:
        
        # We precompute a large voxel array to make it jittable
        voxel_index_array = np.zeros((tsdf_grid.size,3), dtype=np.int64)
        
        voxel_index_count, voxel_index_array = raycast_on_voxel_array_torch(
            voxel_index_array,
            voxel_size,
            tsdf_grid,
            ori,
            pos,
            fx,
            fy,
            cx,
            cy,
            u_min,
            u_max,
            u_step,
            v_min,
            v_max,
            v_step,
            t_min,
            t_max,
            t_step,
            device='cuda'
        )
        voxel_indices = voxel_index_array[:voxel_index_count]

        indices = np.unique(voxel_indices, axis=0)

        bbox_min = ((T_task_base_np[:3,:3] @ self_bbox_min)+T_task_base_np[:3,3]) / voxel_size
        bbox_max = ((T_task_base_np[:3,:3] @ self_bbox_max)+T_task_base_np[:3,3]) / voxel_size
        
        # Get indices of voxels within our pre-defined OBJECT bounding box
        mask = np.all((indices > bbox_min) & (indices < bbox_max), axis=1) # Even faster if jittable with overload
        ig = indices[mask].shape[0]

        if debug:
            i, j, k = indices[mask].T
            tsdfs = tsdf_grid[i, j, k]
            i_j_k_ind = np.vstack((i, j, k))[:, np.logical_and(tsdfs > -1.0, tsdfs < 0.0)].T
            i_j_k_base = []
            for point in i_j_k_ind: i_j_k_base.append(T.apply(point*voxel_size))
            vis.draw_voxels_from_center(i_j_k_base, voxel_size=voxel_size, colors=vis.light_blue)

        igs.append(ig)
    
    return igs

def select_best_grasp(grasps, qualities):
    i = np.argmax(qualities)
    return grasps[i], qualities[i]

def select_data_on_circle(data, goal_angle, go_left):
    # data being angles in [-pi, pi]
    # robot standing at 0
    # go to goal left (decreasing angles) or right (increasing angles)
    # return mask of data that is covered by walking to the goal
    goal_left_of_robot = goal_angle <= 0
    if go_left:
        if goal_left_of_robot:
            data_to_consider = (data <= 0) & (data >= goal_angle)
        else: # robot taking the longer path > pi
            data_to_consider = (data <= 0) | (data >= goal_angle)
    else:
        if not goal_left_of_robot:
            data_to_consider = (data >= 0) & (data <= goal_angle)
        else: # robot taking the longer path > pi
            data_to_consider = (data >= 0) | (data <= goal_angle)

    return data_to_consider

class ActPerMoMaPolicy(Policy):
    def __init__(self, cfg):
        self.qual_threshold = cfg['quality_threshold']
        self.T = cfg['window_size'] # window size
        self.min_z_dist = cfg['min_z_dist']
        self.base_goals_radius_grasp = cfg['base_goals_radius_grasp']
        self.base_goals_radius_ig = cfg['base_goals_radius_ig']
        self.min_path_length = cfg['min_path_length']
        self.max_views = cfg['max_views']
        self.min_gain = cfg['min_gain']
        self.seen_theta_thresh = cfg['seen_theta_thresh']
        self.gain_weighting = cfg['gain_weighting']
        if self.gain_weighting == 'None': self.gain_weighting = None
        self.gain_weighting_factor = cfg['gain_weighting_factor']
        self.gain_discount_fac = cfg['gain_discount_fac']
        self.use_reachability = cfg['use_reachability']
        if self.use_reachability:
            self.reach_eval = ReachabilityEvaluator(arm='both') # both left and right arms
        self.norm_func = cfg['gain_weighting']
        self.arm_activation_radius_base = cfg['arm_activation_radius_base']
        self.arm_activation_radius_goal = cfg['arm_activation_radius_goal']
        self.planner_type = cfg['planner_2d']
        self.fx = cfg['fx']
        self.fy = cfg['fy']
        self.cx = cfg['cx']
        self.cy = cfg['cy']

        self.bbox = None
        self.view_sphere = None
        self.T_base_task = None
        self.T_task_base = None
        self.robot_xy = None
        self.robot_R = None
        self.goal_pos = None
  
        self.last_index = None
        self.last_went_left = None # save the direction chosen at the last step
        self.min_change_mom = cfg['momentum']
        self.deadlock_theta = cfg['deadlock_theta']
        self.steps_along_path = cfg['steps_along_path']

        # defines the generated views
        self.thetas = []
        self.phi = 20 # deg

        # defines the rayscasting steps (in continuous pixel space)
        self.u_step = 1
        self.v_step = 1

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
                                                                [-1.00000000e+00,  5.44363617e-07, -1.08777139e-05,-9.63753913e-04],
                                                                [-2.34190445e-12, -9.98750160e-01, -4.99811739e-02, 1.28999854e+00],
                                                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))

        self.grasp_network = VGN(Path(vgn.__path__[0] + '/../../assets/models/vgn_conv.pth'))
        self.qual_hist = np.zeros((self.T,) + (40,) * 3, np.float32)

        # debug visualization
        self.vis = Visualizer()
        
        # create timeout for pathplanning
        signal.signal(signal.SIGPROF, self.timeout_handler)

        self.activated = False

    def timeout_handler(self, signum, frame):
        signal.setitimer(signal.ITIMER_PROF, 0.0)
        if self.planning:
            print('path planning timeout')
            raise Exception("Timeout!")
    
    def activate(self, goal_pos, goal_aabbox, voxel_size, occ_map, occ_map_cell_size, occ_map_radius):
        self.voxel_size = voxel_size
        
        self.bbox = AABBox(goal_aabbox[0], goal_aabbox[1]) 
        self.view_sphere = ViewHalfSphere(self.bbox, self.min_z_dist)
        self.T_base_task = Transform.from_translation(goal_pos - np.array([0.15, 0.15, 0]))# 0.15 being the tsdf side length 
        self.T_task_base = self.T_base_task.inv()

        self.compute_view_angles()
        self.compute_ray_steps(voxel_size)

        self.views = [] # will be a list of tuples (x, y, theta)
        self.best_grasp = None
        self.nbv = None
        self.aborted = False
        self.last_went_left = None
        self.last_index = None

        # occupancy path planning
        self._occ_map = occ_map
        self._occ_map_cell_size = occ_map_cell_size
        self._occ_map_radius = occ_map_radius
        occ_map_coords = np.flip(np.transpose(np.nonzero(self._occ_map))) # flip because of different co-ordinate system
        s_start, s_goal = (0,0), (1,0) # dummy values initially
        if self.planner_type == 'astar':
            from .search_2D.Astar import AStar
            self.astar = AStar(s_start, s_goal, "euclidean", x_range=self._occ_map.shape[1], y_range=self._occ_map.shape[0], obs_coords=occ_map_coords)
            self.planner_env = self.astar.Env
            # plotting for debugging
            from .search_2D.plotting import Plotting
            self.plot = Plotting(s_start, s_goal, env=self.planner_env)
        elif self.planner_type == 'rrtstar':
            from .sample_2D.rrt_star import RrtStar
            self.rrtstar = RrtStar(s_start, s_goal, step_len=5, goal_sample_rate=0.10, search_radius=5, iter_max=2000, x_range=self._occ_map.shape[1], y_range=self._occ_map.shape[0], obs_coords=occ_map_coords)
        else:
            print('Please set the `params.config.planner_2d` variable in the train .yaml file to `astar` or `rrtstar`')
            raise NotImplementedError

        return True

    def compute_view_angles(self, radius=None):
        """
        goal_z: z coordinate to the center of the bbox
        """
        if radius is None: use_radius=self.view_sphere.r
        else: use_radius = radius
        # for me they are phis, but for Breyer they are thetas, so in my methods its phi, in his its theta
        # 1.42 m is the heighest height the zed camera can take - 1.08 the lowest
        phi_min = np.rad2deg(np.arccos((1.42 - self.bbox.center[2])/use_radius))
        phi_max = np.rad2deg(np.arccos((1.08 - self.bbox.center[2])/use_radius))

        if radius is None: self.thetas = [phi_min, phi_max]
        else: return phi_min, phi_max

    def compute_ray_steps(self, voxel_size):
        bbox_diag = np.sqrt(np.sum(np.square(self.bbox.size)))
        # num_rays u = (self.min_z_dist + bbox_diag)/(self.fx*self.tsdf.voxel_size) and analougly for v
        self.u_step = (self.fx*voxel_size)/(self.min_z_dist + bbox_diag)
        self.v_step = (self.fy*voxel_size)/(self.min_z_dist + bbox_diag)

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
        
        best_base_goal_angles = None
        lefts_or_rights = None
        best_reach_grasp_ids = None
        arms = np.array([0, 0])
        use_grasp_r = False

        ## RESET in different situations
        if (not self.activated) or state['reset_agent']:
            # self.reset_agent_count += 1
            # self.steps_in_episode = 0
            self.activated = self.activate(state['goal_pos'], 
                                           state['goal_aabbox'], 
                                           state['tsdf'].voxel_size,
                                           state['occ_map'],
                                           state['occ_map_cell_size'],
                                           state['occ_map_radius'])
        # else:
        #     self.steps_in_episode += 1  

        if (not self.activated) or state['reset_agent'] or state['reset_theta']:
            self.min_theta = None
            self.max_theta = None
            self.last_theta = None
            self.seen_theta_turns = 0 # handling crossovers at -pi,pi
            self.seen_theta_coverage = 0.0 # total angle covered by views until now
            self.best_grasp = None
            self.nbv = None

        ## UPDATE INFO
        current_view = Transform(translation=state['view_pos'], rotation=self.view_template * Rotation.from_euler('YX', [state['view_theta'], state['view_phi']]))
        self.views.append(current_view)
        # Calculate running view coverage in the XY plane
        self.calculate_theta_coverage(state['view_theta'])
        self.integrate(tsdf)

        ## EVALUATE VIEWS
        views = self.generate_views()
        # Calculate ig per-view
        gains = self.ig_fn_parallel(views, tsdf, debug=False)
        # max over high or low torso for each view angle theta
        self.draw_views(views, gains)
        views, gains, up_or_down = self.max_over_height(views, gains)

        ## DECIDE WHETHER TO USE GRASP OR IG RADIUS//HAVE WE EXPLORED ENOUGH?: if you get closer to the goal (radius_grasp < radius_ig), you won't be able to see the scene fully anymore
        if self.seen_theta_coverage >= self.seen_theta_thresh:
            self.stable_filtered_grasps = self.get_stable_grasps(tsdf, self.filtered_grasps)
            use_grasp_r = len(self.stable_filtered_grasps) > 0
            for grasp in self.stable_filtered_grasps:
                # self.vis.draw_grasps_at(self.T_base_task * grasp.pose, grasp.width)
                self.vis.draw_grasps_at(self.T_base_task * grasp.pose, 0.09) # pal grippers have a max opening of 9 cm)

        ## EVALUATE BASE_GOALS AND THEIR PATHS
        base_goal_candidates, paths_to_base_goals = self.generate_paths(use_grasp_r) # base_goal_cnadidates in goal frame, paths_to_base_goals in world frame
        ig_gains_over_paths, path_lengths = self.ig_length_over_paths(views, gains, base_goal_candidates, paths_to_base_goals)
        # Weight the gains to get ig utilities
        ig_utilities = np.asarray(ig_gains_over_paths) * self.gain_weighting_factor    

        # evaluate reachability
        if self.use_reachability:
            reach_utilities = np.zeros_like(ig_utilities)
            if use_grasp_r:
                reach_utilities, best_reach_grasp_ids, best_base_goal_angles, lefts_or_rights = self.reachability_utilities(base_goal_candidates, self.stable_filtered_grasps)
                if self.gain_weighting is not None:
                    reach_utilities /= path_lengths
                utilities = ig_utilities + reach_utilities
            else:
                utilities = ig_utilities
        else:
            utilities = ig_utilities

        # DEBUG: print ig and reach utilities
        print('ig_utilities: ', ig_utilities)
        if self.use_reachability:
            print('reach_utilities: ', reach_utilities)
        print('')
        
        chosen_index = np.argmax(utilities)
        print('chosen_index: ' + str(chosen_index))
        print(np.max(utilities))

        ## MOMENTUM: Against oscillation/fast direction changes (left or right)
        paths_go_left = self.paths_go_left(base_goal_candidates, paths_to_base_goals)
        momentum = self.min_change_mom if use_grasp_r else self.min_change_mom * self.gain_weighting_factor
        if self.last_went_left is not None:
            now_go_left = paths_go_left[chosen_index]
            if not now_go_left == self.last_went_left and len(utilities[paths_go_left != now_go_left]):
                if sum(paths_go_left != now_go_left) != 0 and not utilities[chosen_index] > np.max(utilities[paths_go_left != now_go_left]) + momentum:
                    help_ind = np.argmax(utilities[paths_go_left != now_go_left])
                    chosen_index = np.argwhere(paths_go_left != now_go_left)[help_ind][0]

        ## DEADLOCK AVOIDANCE: If very similar view_theta (diff < self.deadlock_theta), force to choose different base_goal
        if self.last_theta is not None:
            theta_diff = abs(state['view_theta'] + self.seen_theta_turns*2*np.pi - self.last_theta)
            if theta_diff < self.deadlock_theta and chosen_index == self.last_index and len(utilities) > 1: 
                # remove option of choosing same index as before
                utilities = np.delete(utilities, self.last_index)
                del base_goal_candidates[self.last_index]
                if best_base_goal_angles is not None: best_base_goal_angles = np.delete(best_base_goal_angles, self.last_index)
                if lefts_or_rights is not None: lefts_or_rights = np.delete(lefts_or_rights, self.last_index)
                if best_reach_grasp_ids is not None: best_reach_grasp_ids = np.delete(best_reach_grasp_ids, self.last_index)
                chosen_index = np.argmax(utilities)

        self.last_index = chosen_index
        self.last_went_left = paths_go_left[chosen_index]
        print('chosen_index: ' + str(chosen_index))

        base_goal_candidate = base_goal_candidates[chosen_index]
        self.vis.draw_points(np.hstack((paths_to_base_goals[chosen_index],0.05*np.ones((len(paths_to_base_goals[chosen_index]),1)))), colors='light_green') # NOTE: Uncomment to draw path to chosen base goal! 
        # gain = gains_over_paths[chosen_index]
        
        base_vel = self.get_vel_along_path_to(base_goal_candidate, paths_to_base_goals[chosen_index])

        up_or_down_next = self.up_or_down(up_or_down, views, base_goal_candidate)
        nbv = self.get_nbv_from_base_vel(base_vel, up_or_down_next)
        self.vis.draw_frame_matrix(nbv.as_matrix(), axis_length=0.1, point_size=2)

        # Arm activation:
        # If close enough to chosen base goal:
        # dist_from_rob_to_target_obj = np.linalg.norm(self.goal_pos[:2] - self.robot_xy)
        # if dist_from_rob_to_chosen_base_goal < self.arm_activation_radius_base and dist_from_rob_to_target_obj < self.arm_activation_radius_goal:
        if self.use_reachability and reach_utilities.sum() > 0.0:
            dist_from_rob_to_chosen_base_goal = np.linalg.norm(base_goal_candidate[:2]+self.goal_pos[:2] - self.robot_xy)
            if dist_from_rob_to_chosen_base_goal < self.arm_activation_radius_base:
            # execute grasp: set actions[4]=1, actions[5]=0 for left arm activation
                arms[lefts_or_rights[chosen_index]] = 1
                self.draw_views(None, None)
                self.best_grasp = self.stable_filtered_grasps[best_reach_grasp_ids[chosen_index]]
                best_grasp = cp(self.best_grasp.pose)
                best_grasp.translation = best_grasp.translation + self.T_base_task.translation
                self.vis.draw_grasps_at(self.T_base_task * self.best_grasp.pose, self.best_grasp.width)
                best_base_goal_angle = best_base_goal_angles[chosen_index]
                return best_grasp, (self.seen_theta_coverage, best_base_goal_angle), arms
        elif not self.use_reachability and use_grasp_r:
            dist_from_rob_to_goal = np.linalg.norm(self.goal_pos[:2] - self.robot_xy)
            if dist_from_rob_to_goal < self.arm_activation_radius_goal:
                arms[0] = 1
                self.draw_views(None, None)
                self.best_grasp = self.stable_filtered_grasps[-1]
                best_grasp = cp(self.best_grasp.pose)
                best_grasp.translation = best_grasp.translation + self.T_base_task.translation
                self.vis.draw_grasps_at(self.T_base_task * self.best_grasp.pose, self.best_grasp.width)
                return best_grasp, (self.seen_theta_coverage, 0), arms

        if len(self.views) < self.max_views:
        # if len(self.views) < self.max_views and (reach_utilities.sum() > 0.0 or (self.seen_theta_coverage < 2 * np.pi and (np.max(utilities) > self.min_gain or self.seen_theta_coverage < self.seen_theta_thresh))): # for now, as theta coverage is not fixed    
            # Execute towards next view
            return nbv, (self.seen_theta_coverage, None), arms
        else:
            # no grasps were found after max number of views, OR after going around the whole object OR with too few IG
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

    def get_stable_grasps(self, tsdf, grasps):
        stable_grasps = []
        qual_list = []
        for i, grasp in enumerate(grasps):
            t = grasp.pose.translation
            i, j, k = (t / tsdf.voxel_size).astype(int)              # tsdf grid index of best grasp pose
            qs = self.qual_hist[:, i, j, k]                          # quality history has length T
            if np.count_nonzero(qs) == self.T and np.mean(qs) > self.qual_threshold:
                stable_grasps.append(grasp)
                qual_list.append(np.mean(qs))

        return list(np.array(stable_grasps)[np.argsort(qual_list)]) # ordered in ascending order of quality

    def generate_views(self):
        # here according to breyer is theta and phi notation different then in draw_action/for us
        thetas = np.deg2rad(self.thetas)
        phis = np.arange(360 / self.phi) * np.deg2rad(self.phi)
        view_candidates = []

        for theta, phi in itertools.product(thetas, phis):
            view = self.view_sphere.get_view(theta, phi)
            # make sure x-axis always horizontal, in 'right' direction from the view frame
            view = self.make_suitable_for_mobile_manipulator(view)
            view_candidates.append(view) # NOTE: have removed check for ik

        return view_candidates

    def generate_paths(self, use_grasp_r):
        base_goal_candidates = []
        paths_to_base_goals = None

        radius = self.base_goals_radius_grasp if use_grasp_r else self.base_goals_radius_ig
        phis = np.arange(360 / self.phi) * np.deg2rad(self.phi)
        for phi in phis:
            # Generate goal candidates (base position in the radius of the target object or table)
            # Note: Use goal frame here for simplicity
            x, y = radius * np.cos(phi), radius * np.sin(phi)
            base_goal_candidates.append(np.array([x, y, phi]))

        base_goal_candidates, paths_to_base_goals = self.plan_paths(base_goal_candidates)
        return base_goal_candidates, paths_to_base_goals

    def plan_paths(self, base_goal_candidates, timeout=None):
        feasible_base_goal_candidates = []
        paths_to_base_goals = []
        if timeout is None:
            timeout = 3e-2

        # calculate start pos
        start_xy = self.robot_xy
        s_start = (int((start_xy[0] + self._occ_map_radius) / self._occ_map_cell_size), int((start_xy[1] + self._occ_map_radius) / self._occ_map_cell_size))
        # Make sure start is not in occupancy map
        # NOTE: Make sure the four neighbours of the start COMBINED are also not occupied
        neightbours = [ (s_start[0] + 1, s_start[1] + 1), (s_start[0] - 1, s_start[1] - 1), (s_start[0] + 1, s_start[1] - 1), (s_start[0] - 1, s_start[1] + 1)]
        if self._occ_map[s_start[1], s_start[0]] == 1 \
            or (self._occ_map[neightbours[0][1], neightbours[0][0]] == 1 \
            and self._occ_map[neightbours[1][1], neightbours[1][0]] == 1 \
            and self._occ_map[neightbours[2][1], neightbours[2][0]] == 1 \
            and self._occ_map[neightbours[3][1], neightbours[3][0]] == 1):
            # We are inside the object, so we need to find the closest free cell
            # get robot_xy and phi in goal frame
            robot_pos = self.robot_xy - self.goal_pos[:2] # in_goal_frame
            robot_phi = np.arctan2(robot_pos[1], robot_pos[0]) # in_goal_frame
            # get new xy by moving in the direction of robot_phi until we are outside the object
            robot_pos += self._occ_map_cell_size * np.array([np.cos(robot_phi), np.sin(robot_phi)])
            while self._occ_map[int((robot_pos[1] + self.goal_pos[1] + self._occ_map_radius) / self._occ_map_cell_size), int((robot_pos[0] + self.goal_pos[0] +self._occ_map_radius) / self._occ_map_cell_size)] == 1:
                robot_pos += self._occ_map_cell_size * np.array([np.cos(robot_phi), np.sin(robot_phi)])
            # set new s_start
            s_start = (int((robot_pos[0] + self.goal_pos[0] + self._occ_map_radius) / self._occ_map_cell_size), int((robot_pos[1] + self.goal_pos[1] + self._occ_map_radius) / self._occ_map_cell_size))

        for base_goal_candidate in base_goal_candidates:
            x, y, _ = base_goal_candidate
            # Check for base_goal feasibility i.e. check for overlap with occupancy map
            ## convert base goal to world co-ordinates
            x += self.goal_pos[0]
            y += self.goal_pos[1]
            ## convert base goal to occupancy map co-ordinates
            x_map = int((x + self._occ_map_radius) / self._occ_map_cell_size)
            y_map = int((y + self._occ_map_radius) / self._occ_map_cell_size)
            ## only add base goal if it is not in occupancy map
            # note the order!!; goal not occupied
            if self._occ_map[y_map, x_map] < 1.0:
                # NOTE: Make sure atleast one of the four neighbours of the goalis also not occupied
                neightbours = [ (x_map + 1, y_map + 1), (x_map - 1, y_map - 1), (x_map + 1, y_map - 1), (x_map - 1, y_map + 1)]
                if self._occ_map[neightbours[0][1], neightbours[0][0]] < 1.0 \
                    or self._occ_map[neightbours[1][1], neightbours[1][0]] < 1.0 \
                    or self._occ_map[neightbours[2][1], neightbours[2][0]] < 1.0 \
                    or self._occ_map[neightbours[3][1], neightbours[3][0]] < 1.0:
                    
                    path = None
                    s_goal = (x_map, y_map)
                    
                    ## Plan paths to each feasible base goal
                    if self.planner_type == 'astar':
                        self.astar.s_start = s_start
                        self.astar.s_goal = s_goal
                        self.planning=True
                        signal.setitimer(signal.ITIMER_PROF, timeout)
                        try:
                            path, visited = self.astar.searching()
                            signal.setitimer(signal.ITIMER_PROF, 0.0)
                            self.planning=False
                        except Exception as e:
                            path = None
                            self.planning=False
                        # as sometimes the SIGPROF SIGNAL is delivered delayed, we'll have to catch it here
                        try:
                            self.plot.xI = s_start
                        except Exception as e:
                            self.plot.xI = s_start
                        try:
                            self.plot.xG = s_goal
                        except Exception as e:
                            self.plot.xG = s_goal
                        # self.plot.animation(path, visited, "A*")  # animation  # NOTE: Uncomment to plot paths!
                        try:
                            self.astar.reset()
                        except Exception as e:
                            self.astar.reset()
                        try:
                            old_path = cp(path)
                        except Exception as e:
                            old_path = cp(path)
                    elif self.planner_type == 'rrtstar':
                        from .sample_2D.rrt_star import Node
                        self.rrtstar.s_start = Node(s_start)
                        self.rrtstar.s_goal = Node(s_goal)
                        self.rrtstar.reset()
                        if first_time:
                            first_time = False
                        else:
                            self.rrtstar.plotting.xI = s_start
                            self.rrtstar.plotting.xG = s_goal
                            path = self.rrtstar.planning(plotting=True) # set to true for plotting
                    try:
                        if path is not None:
                            # convert path to world co-ordinates
                            path = np.asarray(path) * self._occ_map_cell_size - self._occ_map_radius
                            # self.vis.draw_points(np.hstack((path,0.05*np.ones((len(path),1))))) # NOTE: Uncomment to draw paths!
                            feasible_base_goal_candidates.append(base_goal_candidate)
                            paths_to_base_goals.append(path)
                    except Exception as e:
                        if old_path is not None:
                            # convert path to world co-ordinates
                            path = np.asarray(old_path) * self._occ_map_cell_size - self._occ_map_radius
                            # self.vis.draw_points(np.hstack((path,0.05*np.ones((len(path),1))))) # NOTE: Uncomment to draw paths!
                            feasible_base_goal_candidates.append(base_goal_candidate)
                            paths_to_base_goals.append(path)

        if len(feasible_base_goal_candidates) < 3:
            return self.plan_paths(base_goal_candidates, timeout=timeout*2)

        return feasible_base_goal_candidates, paths_to_base_goals

    def get_path_length(self, start_point, path, end_point):
        # all in world frame

        airline = np.linalg.norm(start_point - end_point)
        if airline <= self.arm_activation_radius_base:
            airline = 1e-10
        elif airline < self.min_path_length:
            airline = self.min_path_length

        if len(path) <= self.steps_along_path:
            return airline
        
        path_length = 0.0
        # path_length += np.linalg.norm(start_point - path[-1])
        # path_length += np.linalg.norm(end_point - path[0])
        for i in range(len(path)-1):
            path_length += np.linalg.norm(path[i+1] - path[i])

        return path_length

    def ig_length_over_paths(self, views, gains, base_goal_candidates, paths_to_base_goals):
        # views: list of Transforms in world frame
        # gains: list of igs corresponding to views
        # base goal candidates: base goals in goal frame
        # paths_to_base_goals: paths to base goals in world frame, i.e. lists of points [x,y]

        # Get robot's current position to goal frame (invert goal_pos which is in world frame)
        robot_pos = self.robot_xy - self.goal_pos[:2] # in_goal_frame
        # Calculate robot's phi in goal frame (robot is always looking at goal)
        robot_phi = np.arctan2(robot_pos[1], robot_pos[0]) # in_goal_frame

        gains_over_paths = []
        path_lengths = []

        views_phi = []
        for view in views:
            # get view_phi in goal frame
            view_phi = np.arctan2(view.translation[1] - self.goal_pos[1], 
                                view.translation[0] - self.goal_pos[0]) # in_goal_frame
            # set robot_phi to be equal to 0 view_phi
            view_phi -= robot_phi
            # make sure to be in [-pi, pi] range
            if view_phi > np.pi:
                view_phi -= 2*np.pi
            elif view_phi < -np.pi:
                view_phi += 2*np.pi

            views_phi.append(view_phi)
        views_phi = np.array(views_phi)

        if paths_to_base_goals is not None: # some path planner had been used
            for base_goal, path in zip(base_goal_candidates, paths_to_base_goals):
                # set robot_phi to be equal to 0 base_goal_phi
                base_goal_phi = base_goal [2] - robot_phi
                # make to sure to be in [-pi, pi]
                if base_goal_phi > np.pi:
                    base_goal_phi -= 2*np.pi

                goal_left_of_robot = base_goal_phi <= 0

                # check whether path goes to the left or right of the robot
                path_phi = np.arctan2(np.array(path[:,1]) - self.goal_pos[1], np.array(path[:,0]) - self.goal_pos[0]) - robot_phi# in_goal_frame, set robot_phi to be equal to 0
                # make sure to be in [-pi, pi] range
                path_phi[path_phi > np.pi] -= 2*np.pi
                path_phi[path_phi < -np.pi] += 2*np.pi
                if abs(base_goal_phi) > np.deg2rad(self.phi):
                    middle_ind = int(len(path)/2)
                    path_goes_left_of_robot = path_phi[middle_ind] <= 0
                else:
                    path_goes_left_of_robot = goal_left_of_robot

                # select views along path to consider for ig
                views_to_consider = select_data_on_circle(views_phi, base_goal_phi, path_goes_left_of_robot)

                # pathlength to each view following the path
                path_gain = 0.0
                for i, view_phi in enumerate(views_phi[views_to_consider]):
                    # select points along path to follow to reach view
                    path_to_consider = select_data_on_circle(path_phi, view_phi, path_goes_left_of_robot)
                    # define last point in path to view
                    path_phi_to_view = path_phi[path_to_consider] - view_phi
                    if len(path_phi_to_view) > 0:
                        path_phi_to_view[path_phi_to_view > np.pi] -= 2*np.pi
                        path_phi_to_view[path_phi_to_view < -np.pi] += 2*np.pi
                        end_point = path[path_to_consider][np.argmin(abs(path_phi_to_view))]
                        path_length_to_view = self.get_path_length(self.robot_xy, path[path_to_consider], end_point)
                    else:
                        path_length_to_view = np.linalg.norm(self.robot_xy - views[views_to_consider][i].translation[:2])
                    
                    path_length_to_view = max(self.min_path_length, path_length_to_view)

                    if self.gain_weighting is not None:
                        path_gain += gains[views_to_consider][i] / path_length_to_view**2
                    else:
                        path_gain += gains[views_to_consider][i]

                gains_over_paths.append(path_gain)
                path_lengths.append(self.get_path_length(self.robot_xy, path, base_goal[:2] + self.goal_pos[:2]))


        else: # use icecone velocity          
            for base_goal in base_goal_candidates:
                # set robot_phi to be equal to 0 base_goal_phi
                base_goal_phi = base_goal [2] - robot_phi
                # make to sure to be in [-pi, pi]
                if base_goal_phi > np.pi:
                    base_goal_phi -= 2*np.pi

                goal_left_of_robot = base_goal_phi <= 0
                icecone_edge_angle = self.get_icecone_edge_angle_from_robot(robot_pos, ig_state=True)

                if goal_left_of_robot:
                    views_to_consider = (views_phi <= 0) & (views_phi >= base_goal_phi)
                else: # goal is to the right of robot
                    views_to_consider = (views_phi >= 0) & (views_phi <= base_goal_phi)

                path_gain = 0.0
                max_path_dist = self.min_path_length # should be atleast this
                for i, view_phi in enumerate(views_phi[views_to_consider]):
                    goal_to_view = views[views_to_consider][i].translation[:2] - self.goal_pos[:2] # view translation in goal frame
                    goal_to_view_norm = goal_to_view / np.linalg.norm(goal_to_view) # normalize view vector
                    
                    if abs(base_goal_phi) >= icecone_edge_angle: # base goal lies on scoop/circle part of the icecone shortest path to the goal
                        if abs(view_phi) >= icecone_edge_angle: # point to reach this view phi lies on scoop/circle part of the icecone shortest path to the goal
                            dist = self.base_goals_radius_ig * (abs(view_phi) - icecone_edge_angle) # part of the distance that follows the scoop/circle part of the icecone with radius 1
                            dist += self.get_icecone_edge_dist(robot_pos, ig_state=True) # cone part of the icecone shortest path to the base_goal, radius is one
                        
                        else:
                            # define cone edge point
                            cone_edge_point = self.get_icecone_edge_point(robot_pos, goal_left_of_robot, ig_state=True)
                            rob_to_cone_edge = cone_edge_point - robot_pos # vector from robot to cone edge point
                            rob_to_cone_edge_norm = rob_to_cone_edge / np.linalg.norm(rob_to_cone_edge) # normalize 
                            cathedes_length = np.linalg.solve(np.hstack((goal_to_view_norm.reshape(2, 1), 
                                                                        -rob_to_cone_edge_norm.reshape(2, 1))), 
                                                                        robot_pos) # caused a singualr matrix error once
                            dist = abs(cathedes_length[1])
                        
                    else:
                        rob_to_base_goal = base_goal[:2] - robot_pos # robot to base_goal
                        rob_to_base_goal_norm = rob_to_base_goal / np.linalg.norm(rob_to_base_goal) # normalize 
                        cathedes_length = np.linalg.solve(np.hstack((goal_to_view_norm.reshape(2, 1), 
                                                                    -rob_to_base_goal_norm.reshape(2, 1))), 
                                                        robot_pos)
                        dist = abs(cathedes_length[1])

                    # Optional: clip the minimum distance to avoid exploding gains
                    dist = max(self.min_path_length, dist)

                    if self.gain_weighting is not None:
                        path_gain += gains[views_to_consider][i] / dist**2
                    else:
                        path_gain += gains[views_to_consider][i]
                    
                    max_path_dist = max(max_path_dist, dist)
                
                # Store dist for each path for later use
                path_lengths.append(max_path_dist)

                # multiply array with discount factors for waypoints instead of dist
                # path_gain *= self.gain_discount_fac

                gains_over_paths.append(path_gain)
            
        return gains_over_paths, path_lengths

    def paths_go_left(self, base_goals, paths):
        # Get robot's current position to goal frame (invert goal_pos which is in world frame)
        robot_pos = self.robot_xy - self.goal_pos[:2] # in_goal_frame
        # Calculate robot's phi in goal frame (robot is always looking at goal)
        robot_phi = np.arctan2(robot_pos[1], robot_pos[0]) # in_goal_frame
        paths_go_left_of_robot = []

        for base_goal, path in zip(base_goals, paths):
            # set robot_phi to be equal to 0 base_goal_phi
            base_goal_phi = base_goal [2] - robot_phi
            # make to sure to be in [-pi, pi]
            if base_goal_phi > np.pi:
                base_goal_phi -= 2*np.pi

            path_phi = np.arctan2(np.array(path[:,1]) - self.goal_pos[1], np.array(path[:,0]) - self.goal_pos[0]) - robot_phi# in_goal_frame, set robot_phi to be equal to 0
            # make sure to be in [-pi, pi] range
            path_phi[path_phi > np.pi] -= 2*np.pi
            path_phi[path_phi < -np.pi] += 2*np.pi
            middle_ind = int(len(path)/2)
            paths_go_left_of_robot.append(path_phi[middle_ind] <= 0)

        return np.array(paths_go_left_of_robot)

    def make_suitable_for_mobile_manipulator(self, view):
        # view: Transform
        # choose x and y coordinates of the view's new x axis to be orthogonal to z (scalar product = 0), make x-axis go to the right in view direction
        view = view.as_matrix()
        z_y = view[1, 2]
        z_x = view[0, 2]
        
        x_x = 1 * np.sign(z_y) # sign(z_y)
        x_y = -z_x/z_y * np.sign(z_y) if z_y != 0 else -np.sign(z_x) # -z_x / z_y * sign(z_y) if z_y != 0 else - sign (z_x)
        
        x_len = np.sqrt(x_x**2 + x_y**2)
        x_new = np.array([x_x, x_y, 0]) / x_len
        y_new = np.cross(view[:3, 2], x_new)

        view[:3, 0] = x_new
        view[:3, 1] = y_new
        return Transform.from_matrix(view)

    def ig_fn_parallel(self, views, tsdf, debug=False):
        tsdf_grid, voxel_size = tsdf.get_grid(), tsdf.voxel_size
        tsdf_grid = -1.0 + 2.0 * tsdf_grid  # Open3D maps tsdf to [0,1]

        bbox_corners = np.array(self.bbox.corners)
        self_bbox_min = self.bbox.min
        self_bbox_max = self.bbox.max

        T_task_base_np = self.T_task_base.as_matrix()
        view_nps = []
        for view in views:
            view_np = view.as_matrix()
            view_nps.append(view_np)

        igs = ig_fn_parallel_jit(tsdf_grid, voxel_size, self.fx, self.fy, self.cx, self.cy, self.u_step, self.v_step, bbox_corners, self_bbox_min, self_bbox_max, T_task_base_np, view_nps, debug=debug, vis=self.vis, T=self.T_base_task)
        
        if debug:
            self.vis.draw_box_min_max(self.bbox.min, self.bbox.max, point_size=2, color=self.vis.red)
        return igs
    
    def max_over_height(self, views, gains):
        gains = np.array(gains)
        gains_up_down = np.vstack((gains[:int(len(gains)/2)],    # viewpoint further up
                                   gains[int(len(gains)/2):]))   # viewpoint further down
        up_or_down = np.argmax(gains_up_down, axis=0)
        gains = gains_up_down[up_or_down, range(len(up_or_down))]

        views = np.array(views)
        views_up_down = np.vstack((views[:int(len(views)/2)],    # viewpoint further up
                                   views[int(len(views)/2):]))   # viewpoint further down
        views = views_up_down[up_or_down, range(len(up_or_down))]

        return views, gains, up_or_down

    def get_vel_along_path_to(self, base_goal_candidate, path):
        if len(path) <= self.steps_along_path or np.linalg.norm(base_goal_candidate[:2] + self.goal_pos[:2] - self.robot_xy) < self.min_path_length:
            return base_goal_candidate[:2] + self.goal_pos[:2] - self.robot_xy
        
        goal = path[-self.steps_along_path]
        return goal - self.robot_xy

    def up_or_down(self, up_or_down, views, goal_on_circle):
         # Get robot's current position to goal frame (invert goal_pos which is in world frame)
        robot_pos = self.robot_xy - self.goal_pos[:2] # in_goal_frame
        # Calculate robot's phi in goal frame (robot is always looking at goal)
        robot_phi = np.arctan2(robot_pos[1], robot_pos[0]) # in_goal_frame

        views_phi = []
        for view in views:
            # get view_phi in goal frame
            view_phi = np.arctan2(view.translation[1] - self.goal_pos[1], 
                                  view.translation[0] - self.goal_pos[0]) # in_goal_frame
            # set robot_phi to be equal to 0 view_phi
            view_phi -= robot_phi # in_goal_frame
            # make sure to be in [-pi, pi] range
            if view_phi > np.pi:
                view_phi -= 2*np.pi
            elif view_phi < - np.pi:
                view_phi += 2*np.pi

            views_phi.append(view_phi)
        views_phi = np.array(views_phi)

        base_goal_phi = goal_on_circle [2] - robot_phi
        # make to sure to be in [-pi, pi]
        if base_goal_phi > np.pi:
            base_goal_phi -= 2*np.pi

        if base_goal_phi <= 0: # goal is to the left of robot
            views_to_consider = (views_phi <= 0) & (views_phi >= base_goal_phi)
        else: # goal is to the right of robot
            views_to_consider = (views_phi >= 0) & (views_phi <= base_goal_phi)

        if views_phi[views_to_consider].any():
            i = np.argmin(np.abs(views_phi[views_to_consider]))
            return up_or_down[views_to_consider][i]
        else:
            return 1

    def get_nbv_from_base_vel(self, base_vel, up_or_down_next):
        nbv_base_pos = self.robot_xy + base_vel
        nbv_theta = -(np.arctan2(nbv_base_pos[1] - self.goal_pos[1], nbv_base_pos[0] - self.goal_pos[0]) + np.pi)
        base_pose = Transform(translation=np.hstack((nbv_base_pos, np.array([0]))), rotation=Rotation.from_euler('Z', -nbv_theta))

        nbv_pos = base_pose * self.base_to_zed_pose
        nbv_pos.translation[2] = 1.15 if up_or_down_next else 1.35

        nbv_phi = -np.arctan2(np.linalg.norm(nbv_pos.translation[:2] - self.goal_pos[:2]), nbv_pos.translation[2] - self.goal_pos[2])

        nbv = Transform(translation=nbv_pos.translation, rotation=self.view_template * Rotation.from_euler('YX', [nbv_theta, nbv_phi]))
        return nbv

    def reachability_utilities(self, base_goal_candidates, stable_filtered_grasps):
        # Calculate reachability utilities for all base goal candidates, for all base orientations, for all grasps, for all arms!
        utilities = np.zeros(len(base_goal_candidates))

        # get base goals and orientations in world frame:
        base_goal_candidates_np = np.array(base_goal_candidates)
        base_goals_xy_w = base_goal_candidates_np[:,:2] + self.goal_pos[:2]
        base_goals_angles_w = base_goal_candidates_np[:,2] - np.pi # orientation pointing towards goal in world frame
        # compute several candidate base orientations, but pointing in the half sphere towards the goal
        angle_perturbations = np.linspace(-np.pi/2, np.pi/2, num=5, endpoint=True)
        # add all possible base orientations to another dimension
        base_goals_angles_w = np.repeat(base_goals_angles_w[:, np.newaxis], angle_perturbations.shape[0], axis=1)
        base_goals_angles_w += angle_perturbations
        base_goals_angles_w[base_goals_angles_w >  np.pi] -= 2*np.pi # Keep within [-pi, pi]
        base_goals_angles_w[base_goals_angles_w < -np.pi] += 2*np.pi # Keep within [-pi, pi]

        # Get TFs of from base goals to world frame
        base_goals_to_w_tfs = None
        for base_goal_xy_w, base_goal_angle_w in zip(base_goals_xy_w, base_goals_angles_w):
            base_goal_to_w_tf = None
            base_goal_xyz_w = np.hstack((base_goal_xy_w, np.zeros(1)))
            for angle in base_goal_angle_w:
                tfs_over_angles = np.array([Transform(Rotation.from_euler('Z', angle), base_goal_xyz_w).inv().as_matrix()])
                if base_goal_to_w_tf is None:
                    base_goal_to_w_tf = tfs_over_angles
                else:
                    base_goal_to_w_tf = np.vstack((base_goal_to_w_tf, tfs_over_angles))
            
            base_goal_to_w_tf = np.array([base_goal_to_w_tf])
            if base_goals_to_w_tfs is None:
                base_goals_to_w_tfs = base_goal_to_w_tf
            else:
                base_goals_to_w_tfs = np.vstack((base_goals_to_w_tfs, base_goal_to_w_tf))

        stable_grasps = cp(stable_filtered_grasps)
        grasp_tfs_w_frame = None
        for grasp in stable_grasps:
            # convert from TSDF frame to world frame
            # grasp.pose.translation = grasp.pose.translation + self.bbox.center - np.array([0.15, 0.15, 0.1])
            grasp.pose.translation = grasp.pose.translation + self.T_base_task.translation
            if grasp_tfs_w_frame is None:
                grasp_tfs_w_frame = np.array([grasp.pose.as_matrix()])
            else:
                grasp_tfs_w_frame = np.vstack(( grasp_tfs_w_frame, np.array([grasp.pose.as_matrix()]) ))
        
        # get ALL grasp poses in robot base placement frame (base_goals x base_orientations x grasps)
        grasp_poses_in_base_goal_frame = np.matmul(np.expand_dims(base_goals_to_w_tfs,axis=2), grasp_tfs_w_frame)
        
        ## Get reachability scores for all grasps:

        # unroll grasp_poses_in_base_goal_frame
        grasp_poses_in_base_goal_frame = grasp_poses_in_base_goal_frame.reshape(-1, 4, 4)
        left_scores, right_scores = self.reach_eval.get_scores(grasp_poses_in_base_goal_frame, arm='both')
        # re-roll to (base_goals x base_orientations x grasps) dims
        all_left_scores = left_scores.reshape(base_goals_xy_w.shape[0], base_goals_angles_w.shape[1], len(stable_grasps))
        all_right_scores = right_scores.reshape(base_goals_xy_w.shape[0], base_goals_angles_w.shape[1], len(stable_grasps))

        # get max scores over all base_goal_angles
        left_scores_over_angles = np.max(all_left_scores, axis=1)
        right_scores_over_angles = np.max(all_right_scores, axis=1)
        # get max scores over all grasps
        left_scores_over_angles_and_grasps = np.max(left_scores_over_angles, axis=1)
        right_scores_over_angles_and_grasps = np.max(right_scores_over_angles, axis=1)
        # get max scores over left and right arm
        lefts_or_rights = np.argmax(np.vstack((left_scores_over_angles_and_grasps, right_scores_over_angles_and_grasps)), axis=0)

        # get utilities
        utilities = np.max(np.vstack((left_scores_over_angles_and_grasps, right_scores_over_angles_and_grasps)), axis=0)
        # get best grasp ids
        best_reach_grasp_ids = np.argmax(np.where(lefts_or_rights == 0, left_scores_over_angles.T, right_scores_over_angles.T).T, axis=1)
        # get best base goal ids
        left_scores_over_grasps = np.max(all_left_scores, axis=2)
        right_scores_over_grasps = np.max(all_right_scores, axis=2)
        best_base_goal_ids   = np.argmax(np.where(lefts_or_rights == 0, left_scores_over_grasps.T, right_scores_over_grasps.T).T, axis=1)
        best_base_goal_angles = base_goals_angles_w[range(len(base_goals_angles_w)), best_base_goal_ids]
        
        return utilities, best_reach_grasp_ids, best_base_goal_angles, lefts_or_rights
    
    def cost_fn(self, view):
        return 1.0

    def draw_noisy_action(self, state):
        raise NotImplementedError('Tried to draw a noisy action from the ActiveGraspPolicy')

    def integrate(self, tsdf):
        ## from MultiviewPolicy.integrate
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
                top_grasp = pose.rotation.as_matrix()[2, 2] < -np.sqrt(2)/2 # uncomment to only allow side grasps (and no top grasps)
                # if high_enough and not from_below:
                if high_enough and not from_below and not top_grasp:
                    filtered_grasps.append(grasp)
                    filtered_qualities.append(quality)
        return filtered_grasps, filtered_qualities

    def draw_views(self, views, qualities, draw_frames=False):
        # views: list of Transforms in world frame
        # views: list of Transforms in world frame
        # qualities: list of floats
        intrinsics = [self.fx, self.fy, self.cx*2, self.cy*2]

        if views is not None:
            # colors: interpolate between red and green linearly according to quality
            colors = self.vis.interpolate_quality_colors(qualities)

            # draw views to considers in the colors indicating their information gain
            self.vis.draw_views(views, colors, draw_frames=draw_frames, intrinsics=intrinsics)

        # draw the views up until now and the view trajectory taken
        self.vis.draw_views(self.views, [self.vis.light_blue]*len(self.views), point_size = 2, connect_views=True, draw_frames=draw_frames, intrinsics=intrinsics)
      

class ActPerMoMaAgent(Agent):
    """
    Agent that uses the ActiveGraspTrajPolicy to generate a trajectory of views and grasps to execute.
    """
    def __init__(self, mdp_info, agent_params_config):
        self._actor_last_loss = None
        policy = ActPerMoMaPolicy(agent_params_config)

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
