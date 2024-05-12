import numpy as np
from actpermoma.utils.spatial_utils import *

def avoid_collision_2d(current, radius, step, obstacle_bbox, alpha=0.1, beta=5, max_iter=100, is_aabb=False):
    """
    :param current: [x, y] np.ndarray
    :param radius: float, assumed robot radius
    :param step: [x_step, y_step] np.ndarray
    :param obstacle_bboxes: [x_min, y_min, x_max, y_max] np.ndarray (axis-aligned bounding box)
    :return: [x, y] np.ndarray
    """
    ## DO NOT MAKE THE STEP_SIZE LARGER THAN THE RADIUS OF THE ROBOT IF USING THIS FUNCTION

    step_length = np.linalg.norm(step)

    # check for collision with obstacle, if taking step
    new_space = create_circle_points(center=current + step, radius=radius)
    if is_aabb:
        collision_mask = check_collision_2d_aabbox(new_space, obstacle_bbox)
    else:
        collision_mask = check_collision_2d(new_space, obstacle_bbox)

    i = 0
    while sum(collision_mask) > 0 and i < max_iter: # collision detected
        i+=1

        # determine chord of collision
        coll_indices = np.arange(len(new_space))[collision_mask]
        first_coll = coll_indices[0]
        last_coll = coll_indices[-1]
        
        # handle case of collision sequence going over start/end
        if first_coll == 0 and last_coll == len(collision_mask) - 1:
            no_coll_indices = np.arange(len(new_space))[~collision_mask]
            first_coll = no_coll_indices[-1] + 1
            last_coll = no_coll_indices[0] - 1

        chord = new_space[last_coll] - new_space[first_coll]

        # adjust +- if needed
        if np.dot(chord, step) < 0:
            chord = -chord

        # determine depth of chord d
        if is_aabb:
            center_inside = check_collision_2d_aabbox([current], obstacle_bbox)[0] # we'll need this determine the correct case for calculating depth d
        else:
            center_inside = check_collision_2d([current], obstacle_bbox)[0]
        
        if center_inside:
            d = radius + np.sqrt(radius**2 - (np.linalg.norm(chord)/2)**2)
        else:
            d = radius - np.sqrt(radius**2 - (np.linalg.norm(chord)/2)**2)
        
        step = (beta * d + alpha) * chord + (step_length - beta * d - alpha) * step
        # resize it
        step = step/np.linalg.norm(step) * step_length

        # check for collision with obstacle, if taking newly determined step
        new_space = create_circle_points(center=current + step, radius=radius)
        if is_aabb:
            collision_mask = check_collision_2d_aabbox(new_space, obstacle_bbox)
        else:
            collision_mask = check_collision_2d(new_space, obstacle_bbox)

        if d < 0.02:
            break

    return current + step

def check_collision_2d(points, obstacle_bbox, safety_margin=0.1):
    """
    :param points: list of ([x, y] np.ndarray)
    :param obstacle_bbox: [x_min, y_min, x_max, y_max, z_to_ground, yaw] np.ndarray (bounding box)
    :return: list of bool, length of points
    """
    # create bbox frame (add safety margin) 
    obstacle_rot = Rotation.from_euler('z', obstacle_bbox[5])
    safety_margin_applied = obstacle_rot.apply(np.array([safety_margin/2, safety_margin/2, 0]))
    obstacle_frame = Transform(translation=np.hstack((obstacle_bbox[:2], [0])) - safety_margin_applied,
                            rotation=obstacle_rot)
    
    obstacle_frame_inv = obstacle_frame.inv()

    max_xy = obstacle_frame_inv.apply(np.array([obstacle_bbox[2], obstacle_bbox[3], 0]))[:2] + np.array([safety_margin/2, safety_margin/2])
    
    collision_mask = []
    for point in points:
        point = np.hstack((point, [0]))
        point = obstacle_frame_inv.apply(point)
        collision_mask.append(0 <= point[0] <= max_xy[0] and 0 <= point[1] <= max_xy[1])

    return np.array(collision_mask)
    
def check_collision_2d_aabbox(points, obstacle_aabbox):
    """
    :param points: list of ([x, y] np.ndarray)
    :param obstacle_aabbox: [x_min, y_min, x_max, y_max] np.ndarray (axis-aligned bounding box)
    :return: list of bool, length of points
    """
    return np.array([obstacle_aabbox[0] <= point[0] <= obstacle_aabbox[2] and \
            obstacle_aabbox[1] <= point[1] <= obstacle_aabbox[3] for point in points])

def create_circle_points(center, radius, num=100):
    theta = np.linspace(0, 2*np.pi, num=num)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.vstack((x, y)).T