import numpy as np
import torch
import itertools
from .numba_overloads import nb_unique
from copy import deepcopy as cp
import signal

from mushroom_rl.policy import Policy
from mushroom_rl.core import Agent

import vgn
from pathlib import Path
from vgn.detection import VGN, select_local_maxima

from actpermoma.utils.visualisation_utils import Visualizer
from actpermoma.utils.algo_utils import AABBox, ViewHalfSphere

from robot_helpers.spatial import Rotation, Transform

"""
This is the active-grasp policy following this paper [0], but was adjusted for mobile manipulation.
[0]: Breyer, Michel, et al. "Closed-loop next-best-view planning for target-driven grasping." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.
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

        # indices = np.unique(voxel_indices, axis=0)
        indices = nb_unique(voxel_indices, axis=0)[0] # Jittable unique function from https://github.com/numba/numba/issues/7663#issuecomment-999196241

        bbox_min = ((T_task_base_np[:3,:3] @ self_bbox_min)+T_task_base_np[:3,3]) / voxel_size
        bbox_max = ((T_task_base_np[:3,:3] @ self_bbox_max)+T_task_base_np[:3,3]) / voxel_size
        
        # Get indices of voxels within our pre-defined OBJECT bounding box
        mask = np.all((indices > bbox_min) & (indices < bbox_max), axis=1) # Even Faster because jittable with overload
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


class ActiveGraspPolicy(Policy):
    def __init__(self, cfg):
        self.qual_threshold = cfg['quality_threshold']
        self.T = cfg['window_size'] # window size
        self.min_z_dist = cfg['min_z_dist']
        self.base_goals_radius_grasp = cfg['base_goals_radius_grasp']
        self.base_goals_radius_ig = cfg['base_goals_radius_ig']
        self.min_path_length = cfg['min_path_length']
        self.max_views = cfg['max_views']
        self.min_gain = cfg['min_gain']
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
        self.best_grasp_base_goal = None

        self.grasp_exec = None
        self.best_grasp = None
        self.filtered_grasps = []
        self.stable_filtered_grasps = []
        self.nbv = None
        self.aborted = False

        self.base_to_zed_pose = Transform.from_matrix(np.array([[-1.08913264e-05, -4.99811739e-02,  9.98750160e-01, 2.12868273e-01],
                                                                [-1.00000000e+00,  5.44363617e-07, -1.08777139e-05,-9.63753913e-04],
                                                                [-2.34190445e-12, -9.98750160e-01, -4.99811739e-02, 1.38999854e+00],
                                                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]))

        self.grasp_network = VGN(Path(vgn.__path__[0] + '/../../assets/models/vgn_conv.pth'))
        self.qual_hist = np.zeros((self.T,) + (40,) * 3, np.float32)

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
        self.view_sphere = ViewHalfSphere(self.bbox, self.base_goals_radius_ig)
        box_height = goal_aabbox[1,2] - goal_aabbox[0,2]
        self.T_base_task = Transform.from_translation(goal_pos - np.array([0.15, 0.15, 0]))# 0.15 being the tsdf side length
        self.T_task_base = self.T_base_task.inv()

        self.compute_view_angles()
        self.compute_ray_steps(voxel_size)

        self.views = [] # will be a list of tuples (x, y, theta)
        self.best_grasp = None
        self.nbv = None
        self.aborted = False
        self.grasp_exec = False
        self.best_grasp_base_goal = None

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
        
        if (not self.activated) or state['reset_agent']:
            self.activated = self.activate(state['goal_pos'], 
                                           state['goal_aabbox'], 
                                           state['tsdf'].voxel_size,
                                           state['occ_map'],
                                           state['occ_map_cell_size'],
                                           state['occ_map_radius'])

        arms = np.array([0, 0])
        # calculate current view
        current_view = Transform(translation=state['view_pos'], rotation=self.view_template * Rotation.from_euler('YX', [state['view_theta'], state['view_phi']]))
        self.views.append(current_view)

        # Calculate running view coverage in the XY plane
        self.calculate_theta_coverage(state['view_theta'])

        if len(self.views) >= self.max_views or self.best_grasp_prediction_is_stable(tsdf):
            self.grasp_exec = True

        self.integrate(tsdf)
        views = self.generate_views() # views in world space
        
        # Calculate ig per-view
        gains = self.ig_fn_parallel(views, tsdf, debug=False)
        # print('gains: ', gains)
        
        # max over high or low torso for each view angle theta
        all_views = cp(views); all_gains = cp(gains)
        views, gains, up_or_down = self.max_over_height(views, gains)

        i = np.argmax(gains)
        nbv, gain, up_or_down_next = views[i], gains[i], up_or_down[i]
        self.vis.draw_frame_matrix(nbv.as_matrix(), axis_length=0.1, point_size=2)
        if self.grasp_exec and self.best_grasp is not None and np.linalg.norm(self.goal_pos[:2] - self.robot_xy) < self.arm_activation_radius_goal:
            arms[0] = 1
            self.draw_views(None, None)
            best_grasp = cp(self.best_grasp.pose)
            best_grasp.translation = best_grasp.translation + self.T_base_task.translation
            self.vis.draw_grasps_at(self.T_base_task * self.best_grasp.pose, self.best_grasp.width)
            return best_grasp, (self.seen_theta_coverage, 0), arms
        elif (gain >= self.min_gain or len(self.views) <= self.T):
            # DEBUG: print ig and reach utilities
            print('ig_utilities: ', gains)
            print('')

            self.draw_views(all_views, all_gains)
        
            base_goal_candidate = nbv.translation[:2] - self.goal_pos[:2]
            # try to get feasible path and
            feasible_base_goals, paths_to_base_goal = self.plan_paths([base_goal_candidate], timeout=1e-1)
            while len(feasible_base_goals)== 0:
                views = np.delete(views, i)
                gains = np.delete(gains, i)
                if len(views) == 0:
                    self.aborted = True
                    return None, (self.seen_theta_coverage, None), arms
                i = np.argmax(gains)
                nbv, gain, up_or_down_next = views[i], gains[i], up_or_down[i]
                base_goal_candidate = nbv.translation[:2] - self.goal_pos[:2]
                feasible_base_goals, paths_to_base_goal = self.plan_paths([base_goal_candidate], timeout=1e-1)
            base_goal_candidate = feasible_base_goals[0]
            path_to_base_goal = paths_to_base_goal[0]

            base_vel = self.get_vel_along_path_to(base_goal_candidate, path_to_base_goal)
            nbv = self.get_nbv_from_base_vel(base_vel, up_or_down_next)
            return nbv, (self.seen_theta_coverage, None), arms
        else: # no stable grasp found
            self.aborted = True
            return None, (self.seen_theta_coverage, None), arms

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
            x, y  = base_goal_candidate
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

        return feasible_base_goal_candidates, paths_to_base_goals

    def get_vel_along_path_to(self, base_goal_candidate, path):
        if len(path) <= self.steps_along_path or np.linalg.norm(base_goal_candidate[:2] + self.goal_pos[:2] - self.robot_xy) < self.min_path_length:
            return base_goal_candidate[:2] + self.goal_pos[:2] - self.robot_xy
        
        goal = path[-self.steps_along_path]
        return goal - self.robot_xy

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
        for i in range(len(path)-1):
            path_length += np.linalg.norm(path[i+1] - path[i])

        return path_length

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
        for i, grasp in enumerate(grasps):
            t = grasp.pose.translation
            i, j, k = (t / tsdf.voxel_size).astype(int)              # tsdf grid index of best grasp pose
            qs = self.qual_hist[:, i, j, k]                          # quality history has length T
            if np.count_nonzero(qs) == self.T and np.mean(qs) > self.qual_threshold:
                stable_grasps.append(grasp)

        return stable_grasps
    
    def best_grasp_prediction_is_stable(self, tsdf):
        if self.best_grasp is not None:
            t = self.best_grasp.pose.translation
            i, j, k = (t / tsdf.voxel_size).astype(int)          # tsdf grid index of best grasp pose
            qs = self.qual_hist[:, i, j, k]                           # quality history has length T
            if np.count_nonzero(qs) == self.T and np.mean(qs) > self.qual_threshold:
                return True
        return False

    def generate_views(self): # current configuration
        # here according to breyer is theta and phi notation different then in draw_action/for us
        thetas = np.deg2rad(self.thetas)
        phis = np.arange(360 / self.phi) * np.deg2rad(self.phi)
        view_candidates = []
        for theta, phi in itertools.product(thetas, phis):
            if not self.grasp_exec:
                view = self.view_sphere.get_view(theta, phi)
            else:
                view = ViewHalfSphere(self.bbox, self.base_goals_radius_grasp+0.1).get_view(theta, phi)
            # make sure x-axis always horizontal, in 'right' direction from the view frame
            view = self.make_suitable_for_mobile_manipulator(view)
            view_candidates.append(view) # NOTE: have removed check for ik
        
        return view_candidates

    def make_suitable_for_mobile_manipulator(self, view):
        # view: Transform
        # choose x and y coordinates of the view's new x axis to be orthogonal to z (scalar product = 0), make x-axis go to the right in view direction
        view = view.as_matrix()
        z_y = view[1, 2]
        z_x = view[0, 2]
        
        x_x = np.sign(z_y)
        x_y = -z_x/z_y * np.sign(z_y) if z_y != 0 else -np.sign(z_x)
        
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
        nbv_pos.translation[2] = 1.08 if up_or_down_next else 1.42

        nbv_phi = -np.arctan2(np.linalg.norm(nbv_pos.translation[:2] - self.goal_pos[:2]), nbv_pos.translation[2] - self.goal_pos[2])

        nbv = Transform(translation=nbv_pos.translation, rotation=self.view_template * Rotation.from_euler('YX', [nbv_theta, nbv_phi]))
        return nbv

    def cost_fn(self, view):
        return 1.0

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
      

class ActiveGraspAgent(Agent):
    """
    Agent that uses the ActiveGraspTrajPolicy to generate a trajectory of views and grasps to execute.
    """
    def __init__(self, mdp_info, agent_params_config):
        self._actor_last_loss = None
        policy = ActiveGraspPolicy(agent_params_config)

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