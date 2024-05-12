import os
import pickle
import numpy as np
from scipy import Rotation


########################################################################
# Reachability score evaluator
# (based on https://github.com/iROSA-lab/sampled_reachability_maps)
# Author: Snehal Jauhri, 2023
########################################################################

class ReachabilityEvaluator:
    def __init__(self, reach_map_path=None, arm='both', cartesian_res=0.05, angular_res=np.pi / 8):
        if reach_map_path is None:
            reach_map_path = os.path.dirname(os.path.realpath(__file__))
        self.arm = arm
        self.cartesian_res = cartesian_res
        self.angular_res = angular_res

        # Load reachability maps
        if arm == 'right' or arm == 'both':
            with open(os.path.join(reach_map_path,
                                   'smaller_full_reach_map_gripper_right_grasping_frame_torso_False_0.05.pkl'),
                      'rb') as f:
                self.right_reachability_map = pickle.load(f)
        if arm == 'left' or arm == 'both':
            with open(os.path.join(reach_map_path,
                                   'smaller_full_reach_map_gripper_left_grasping_frame_torso_False_0.05.pkl'),
                      'rb') as f:
                self.left_reachability_map = pickle.load(f)

    def get_scores(self, goal_tfs, arm='both', return_max=False):
        # Get reachability scores for 6D goal poses

        left_scores = np.zeros((len(goal_tfs)))
        right_scores = np.zeros((len(goal_tfs)))

        goal_poses = np.zeros((len(goal_tfs), 6))
        for id, goal_tf in enumerate(goal_tfs):
            # Get the 6D euler goal pose
            goal_poses[id] = np.hstack(
                (goal_tf[:3, -1], Rotation.from_matrix(goal_tf[:3, :3]).as_euler('XYZ')))  # INTRINSIC XYZ

        if arm == 'left' or arm == 'both':
            min_y, max_y, = (-0.6, 1.35)
            min_x, max_x, = (-1.2, 1.2)
            min_z, max_z, = (-0.35, 2.1)
            min_roll, max_roll, = (-np.pi, np.pi)
            min_pitch, max_pitch, = (-np.pi / 2, np.pi / 2)
            min_yaw, max_yaw, = (-np.pi, np.pi)
            cartesian_res = self.cartesian_res
            angular_res = self.angular_res

            # Mask valid goal_poses that are inside the min-max xyz bounds of the reachability map
            mask = np.logical_and.reduce((goal_poses[:, 0] > min_x, goal_poses[:, 0] < max_x,
                                          goal_poses[:, 1] > min_y, goal_poses[:, 1] < max_y,
                                          goal_poses[:, 2] > min_z, goal_poses[:, 2] < max_z))

            x_bins = np.ceil((max_x - min_x) / cartesian_res)
            y_bins = np.ceil((max_y - min_y) / cartesian_res)
            z_bins = np.ceil((max_z - min_z) / cartesian_res)
            roll_bins = np.ceil((max_roll - min_roll) / angular_res)
            pitch_bins = np.ceil((max_pitch - min_pitch) / angular_res)
            yaw_bins = np.ceil((max_yaw - min_yaw) / angular_res)

            # Define the offset values for indexing the map
            x_ind_offset = y_bins * z_bins * roll_bins * pitch_bins * yaw_bins
            y_ind_offset = z_bins * roll_bins * pitch_bins * yaw_bins
            z_ind_offset = roll_bins * pitch_bins * yaw_bins
            roll_ind_offset = pitch_bins * yaw_bins
            pitch_ind_offset = yaw_bins
            yaw_ind_offset = 1

            # Convert the input pose to voxel coordinates
            x_idx = (np.floor((goal_poses[mask, 0] - min_x) / cartesian_res)).astype(int)
            y_idx = (np.floor((goal_poses[mask, 1] - min_y) / cartesian_res)).astype(int)
            z_idx = (np.floor((goal_poses[mask, 2] - min_z) / cartesian_res)).astype(int)
            roll_idx = (np.floor((goal_poses[mask, 3] - min_roll) / angular_res)).astype(int)
            pitch_idx = (np.floor((goal_poses[mask, 4] - min_pitch) / angular_res)).astype(int)
            yaw_idx = (np.floor((goal_poses[mask, 5] - min_yaw) / angular_res)).astype(int)
            # Handle edge cases of discretization (angles can especially cause issues if values contain both ends [-pi, pi] which we don't want
            roll_idx = np.clip(roll_idx, 0, roll_bins - 1)
            pitch_idx = np.clip(pitch_idx, 0, pitch_bins - 1)
            yaw_idx = np.clip(yaw_idx, 0, yaw_bins - 1)

            # Compute the index in the reachability map array
            map_idx = x_idx * x_ind_offset + y_idx * y_ind_offset + z_idx * z_ind_offset + roll_idx \
                      * roll_ind_offset + pitch_idx * pitch_ind_offset + yaw_idx * yaw_ind_offset

            # Get the score from the score map array
            left_scores[mask] = self.left_reachability_map[map_idx.astype(int), -1]  # -1 is the score index

        if arm == 'right' or arm == 'both':
            min_y, max_y, = (-1.35, 0.6)
            min_x, max_x, = (-1.2, 1.2)
            min_z, max_z, = (-0.35, 2.1)
            min_roll, max_roll, = (-np.pi, np.pi)
            min_pitch, max_pitch, = (-np.pi / 2, np.pi / 2)
            min_yaw, max_yaw, = (-np.pi, np.pi)
            cartesian_res = self.cartesian_res
            angular_res = self.angular_res

            # Mask valid goal_poses that are inside the min-max xyz bounds of the reachability map
            mask = np.logical_and.reduce((goal_poses[:, 0] > min_x, goal_poses[:, 0] < max_x,
                                          goal_poses[:, 1] > min_y, goal_poses[:, 1] < max_y,
                                          goal_poses[:, 2] > min_z, goal_poses[:, 2] < max_z))

            x_bins = np.ceil((max_x - min_x) / cartesian_res)
            y_bins = np.ceil((max_y - min_y) / cartesian_res)
            z_bins = np.ceil((max_z - min_z) / cartesian_res)
            roll_bins = np.ceil((max_roll - min_roll) / angular_res)
            pitch_bins = np.ceil((max_pitch - min_pitch) / angular_res)
            yaw_bins = np.ceil((max_yaw - min_yaw) / angular_res)

            # Define the offset values for indexing the map
            x_ind_offset = y_bins * z_bins * roll_bins * pitch_bins * yaw_bins
            y_ind_offset = z_bins * roll_bins * pitch_bins * yaw_bins
            z_ind_offset = roll_bins * pitch_bins * yaw_bins
            roll_ind_offset = pitch_bins * yaw_bins
            pitch_ind_offset = yaw_bins
            yaw_ind_offset = 1

            # Convert the input pose to voxel coordinates
            x_idx = (np.floor((goal_poses[mask, 0] - min_x) / cartesian_res)).astype(int)
            y_idx = (np.floor((goal_poses[mask, 1] - min_y) / cartesian_res)).astype(int)
            z_idx = (np.floor((goal_poses[mask, 2] - min_z) / cartesian_res)).astype(int)
            roll_idx = (np.floor((goal_poses[mask, 3] - min_roll) / angular_res)).astype(int)
            pitch_idx = (np.floor((goal_poses[mask, 4] - min_pitch) / angular_res)).astype(int)
            yaw_idx = (np.floor((goal_poses[mask, 5] - min_yaw) / angular_res)).astype(int)
            # Handle edge cases of discretization (angles can especially cause issues if values contain both ends [-pi, pi] which we don't want
            roll_idx = np.clip(roll_idx, 0, roll_bins - 1)
            pitch_idx = np.clip(pitch_idx, 0, pitch_bins - 1)
            yaw_idx = np.clip(yaw_idx, 0, yaw_bins - 1)

            # Compute the index in the reachability map array
            map_idx = x_idx * x_ind_offset + y_idx * y_ind_offset + z_idx * z_ind_offset + roll_idx \
                      * roll_ind_offset + pitch_idx * pitch_ind_offset + yaw_idx * yaw_ind_offset

            # Get the score from the score map array
            right_scores[mask] = self.right_reachability_map[map_idx.astype(int), -1]  # -1 is the score index

        if arm == 'both':
            if return_max:
                return np.maximum(left_scores, right_scores)
            else:
                return left_scores, right_scores
        elif arm == 'left':
            return left_scores
        elif arm == 'right':
            return right_scores
