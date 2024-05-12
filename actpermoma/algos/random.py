import numpy as np
from numba import jit
from copy import deepcopy as cp
import signal
from pathlib import Path

from mushroom_rl.policy import Policy
from mushroom_rl.core import Agent

import vgn
from vgn.detection import VGN, select_local_maxima

from actpermoma.utils.visualisation_utils import Visualizer
from actpermoma.utils.algo_utils import AABBox, ViewHalfSphere

from robot_helpers.spatial import Rotation, Transform

"""
this is the random policy, randomly choosing from the identified feasible paths.
"""


@jit(nopython=True)
def get_voxel_at(voxel_size, p):
    index = (p / voxel_size).astype(np.int64)
    return index if (index >= 0).all() and (index < 40).all() else None


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


class RandomPolicy(Policy):
    def __init__(self, cfg):
        self.qual_threshold = cfg['quality_threshold']
        self.T = cfg['window_size'] # window size
        self.min_z_dist = cfg['min_z_dist']
        self.base_goals_radius_grasp = cfg['base_goals_radius_grasp']
        self.base_goals_radius_ig = cfg['base_goals_radius_ig']
        self.min_path_length = cfg['min_path_length']
        self.max_views = cfg['max_views']
        self.seen_theta_thresh = cfg['seen_theta_thresh']
        self.arm_activation_radius_base = cfg['arm_activation_radius_base']
        self.arm_activation_radius_goal = cfg['arm_activation_radius_goal']
        self.planner_type = cfg['planner_2d']

        self.bbox = None
        self.view_sphere = None
        self.T_base_task = None
        self.T_task_base = None
        self.robot_xy = None
        self.robot_R = None
        self.goal_pos = None
  
        self.last_index = None
        self.last_went_left = None # save the direction chosen at the last step
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
                                                                [-1.00000000e+00,  5.44363617e-07, -1.08777139e-05, -9.63753913e-04],
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

        arms = np.array([0, 0])
        use_grasp_r = False

        ## RESET in different situations
        if (not self.activated) or state['reset_agent']:
            self.activated = self.activate(state['goal_pos'], 
                                           state['goal_aabbox'], 
                                           state['tsdf'].voxel_size,
                                           state['occ_map'],
                                           state['occ_map_cell_size'],
                                           state['occ_map_radius'])

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

        ## DECIDE WHETHER TO USE GRASP OR IG RADIUS//HAVE WE EXPLORED ENOUGH?: if you get closer to the goal (radius_grasp < radius_ig), you won't be able to see the scene fully anymore
        if self.seen_theta_coverage >= self.seen_theta_thresh:
            self.stable_filtered_grasps = self.get_stable_grasps(tsdf, self.filtered_grasps)
            use_grasp_r = len(self.stable_filtered_grasps) > 0
            for grasp in self.stable_filtered_grasps:
                self.vis.draw_grasps_at(self.T_base_task * grasp.pose, 0.09) # pal grippers have a max opening of 9 cm)

        base_goal_candidates, paths_to_base_goals = self.generate_paths(use_grasp_r) # base_goal_cnadidates in goal frame, paths_to_base_goals in world frame

        chosen_index = np.random.randint(len(base_goal_candidates))

        print('chosen_index: ' + str(chosen_index))

        base_goal_candidate = base_goal_candidates[chosen_index]
        # NOTE: Uncomment to draw path to chosen base goal! 
        self.vis.draw_points(np.hstack((paths_to_base_goals[chosen_index],0.05*np.ones((len(paths_to_base_goals[chosen_index]),1)))), colors='light_green') 
        
        base_vel = self.get_vel_along_path_to(base_goal_candidate, paths_to_base_goals[chosen_index])

        up_or_down_next = np.random.randint(2) 
        nbv = self.get_nbv_from_base_vel(base_vel, up_or_down_next)
        self.vis.draw_frame_matrix(nbv.as_matrix(), axis_length=0.1, point_size=2)

        # Arm activation:
        # If close enough to chosen base goal:
        if use_grasp_r:
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

        if self.planner_type is not None:
            base_goal_candidates, paths_to_base_goals = self.plan_paths(base_goal_candidates)
        return base_goal_candidates, paths_to_base_goals

    def plan_paths(self, base_goal_candidates, timeout=None):
        feasible_base_goal_candidates = []
        paths_to_base_goals = []
        if timeout is None:
            timeout = 1e-1 # 3e-2

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
      

class RandomAgent(Agent):
    """
    Agent that uses the ActiveGraspTrajPolicy to generate a trajectory of views and grasps to execute.
    """
    def __init__(self, mdp_info, agent_params_config):
        self._actor_last_loss = None
        policy = RandomPolicy(agent_params_config)

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
