# from typing import Optional
# import numpy as np
import os
import numpy as np
import torch
from pxr import Gf, UsdPhysics
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.utils.semantics import add_update_semantics
from scipy.spatial.transform import Rotation
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from actpermoma.utils.files import get_usd_path

# collision boxes
import shapely
from shapely import geometry

# Utility functions to build a scene with obstacles and objects to be grasped (grasp objects)


def spawn_obstacle(self, name, prim_path, device):
    # Spawn Shapenet obstacle model from usd path
    object_usd_path = os.path.join(get_usd_path(),'Props','Shapenet',name,'models','model_normalized.usd')
    add_reference_to_stage(object_usd_path, prim_path + "/obstacle/" + name)

    obj = GeometryPrim(
        prim_path=prim_path + "/obstacle/" + name,
        name=name,
        position= torch.tensor([0.0, 0.0, 0.0], device=device),
        orientation= torch.tensor([0.707106, 0.707106, 0.0, 0.0], device=device), # Shapenet model may be downward facing. Rotate in X direction by 90 degrees,
        visible=True,
        scale=[0.01,0.01,0.01], # Has to be scaled down to metres. Default usd units for these objects is cms
        collision=True
    )
    # Attach rigid body and enable tight collision approximation
    obj.set_collision_approximation("convexDecomposition")
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(obj.prim_path))
    rigid_api.CreateRigidBodyEnabledAttr(True)

    # Add semantic information to obstacle
    add_update_semantics(obj.prim, name)

    return obj

def spawn_grasp_object(self, name, prim_path, device):
    # Spawn YCB object model from usd path
    object_usd_path = os.path.join(get_usd_path(),'Props','YCB','Axis_Aligned',name+'.usd')
    # object_usd_path = os.path.join(get_usd_path(),'Props','Blocks',name+'.usd')
    add_reference_to_stage(object_usd_path, prim_path + "/grasp_obj/ycb_" + name)

    obj = GeometryPrim(
        prim_path=prim_path + "/grasp_obj/ycb_" + name,
        name=name,
        position= torch.tensor([0.0, 0.0, 0.0], device=device),
        orientation= torch.tensor([0.707106, -0.707106, 0.0, 0.0], device=device), # YCB model may be downward facing. Rotate in X direction by -90 degrees,
        visible=True,
        scale=[0.01,0.01,0.01], # Has to be scaled down to metres. Default usd units for these objects is cms
        collision=True
    )
    
    # Attach rigid body and enable tight collision approximation
    obj.set_collision_approximation("convexDecomposition")
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(obj.prim_path))
    rigid_api.CreateRigidBodyEnabledAttr(True)

    # Add semantic information to object
    add_update_semantics(obj.prim, name)

    return obj

def setup_tabular_scene(self, scene_type, obstacles, tabular_obstacle_mask, grasp_objs, obstacles_dimensions, grasp_objs_dimensions, world_xy_radius, device, return_obstacles=False, return_goal_aabbox=False, occ_map_radius=4.0, occ_map_cell_size=0.1):
    # Randomly arrange the objects in the environment. Ensure no overlaps, collisions!
    # Grasp objects will be placed on a random tabular obstacle.
    # Returns: target grasp object, target grasp/goal, all object's oriented bboxes
    object_positions, object_yaws, objects_dimensions = [], [], []
    obst_aabboxes, grasp_obj_aabboxes = [], []
    obst_collision_2D_boxes = []
    obst_collision_expansion = 0.32 # 0.25 m expansion for collision avoidance
    robot_radius = 0.45 # metres. To exclude circle at origin where the robot (Tiago) is

    # Choose one tabular obstacle to place grasp objects on
    tab_index = np.random.choice(np.nonzero(tabular_obstacle_mask)[0])
    # Place tabular obstacle at random location on the ground plane
    tab_xyz_size = obstacles_dimensions[tab_index][1] - obstacles_dimensions[tab_index][0]
    tab_z_to_ground = - obstacles_dimensions[tab_index][0,2]
    # polar co-ords
    tab_r = world_xy_radius
    tab_phi = np.random.uniform(-np.pi,np.pi)
    tab_x, tab_y = tab_r*np.cos(tab_phi), tab_r*np.sin(tab_phi)
    tab_position = [tab_x,tab_y,tab_z_to_ground]
    tab_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
    obstacles[tab_index].set_world_pose(position=torch.tensor(tab_position,dtype=torch.float,device=device))

    
    # Place all grasp objects on the tabular obstacle (without overlaps)
    for idx, _ in enumerate(grasp_objs):
        grasp_obj_z_to_ground = - grasp_objs_dimensions[idx][0,2] # TODO check if correct
        while(1): # Be careful about infinite loops!
            # Add random orientation (yaw) to object
            grasp_obj_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
            grasp_object_orientation = Rotation.from_euler("xyz", [torch.pi/2,0,grasp_obj_yaw], degrees=False).as_quat()[np.array([3,0,1,2])]
            # Place object at height of tabular obstacle and in the x-y range of the tabular obstacle, within 50 x 50 cm suqare in the middle of the table
            if scene_type == 'complex':
                xy_range = 0.4
            else:
                xy_range = 0.25
            grasp_obj_x = tab_x + np.random.uniform((-xy_range-grasp_objs_dimensions[idx][0,0])/2.0, (xy_range-grasp_objs_dimensions[idx][1,0])/2.0)
            grasp_obj_y = tab_y + np.random.uniform((-xy_range-grasp_objs_dimensions[idx][0,1])/2.0, (xy_range-grasp_objs_dimensions[idx][1,1])/2.0)
            grasp_obj_z = tab_xyz_size[2] + grasp_obj_z_to_ground # Place on top of tabular obstacle
            grasp_obj_position = [grasp_obj_x,grasp_obj_y,grasp_obj_z]
            grasp_objs[idx].set_world_pose(position=torch.tensor(grasp_obj_position,dtype=torch.float,device=device),
                                           orientation=torch.tensor(grasp_object_orientation,dtype=torch.float,device=device)) # YCB needs X -90 deg rotation
            # compute new AxisAligned bbox
            self._scene._bbox_cache.Clear()
            grasp_obj_aabbox = self._scene.compute_object_AABB(grasp_objs[idx].name)
            # Check for overlap with all existing grasp objects
            overlap = False
            for other_aabbox in grasp_obj_aabboxes: # loop over existing AAbboxes
                grasp_obj_range = Gf.Range3d(Gf.Vec3d(grasp_obj_aabbox[0,0],grasp_obj_aabbox[0,1],grasp_obj_aabbox[0,2]),Gf.Vec3d(grasp_obj_aabbox[1,0],grasp_obj_aabbox[1,1],grasp_obj_aabbox[1,2]))
                other_obj_range = Gf.Range3d(Gf.Vec3d(other_aabbox[0,0],other_aabbox[0,1],other_aabbox[0,2]),Gf.Vec3d(other_aabbox[1,0],other_aabbox[1,1],other_aabbox[1,2]))
                intersec = Gf.Range3d.GetIntersection(grasp_obj_range, other_obj_range)
                if (not intersec.IsEmpty()):
                    overlap = True # Failed. Try another pose
                    break
            if (overlap):
                continue # Failed. Try another pose
            else:
                # Success. Add this valid AAbbox to the list
                grasp_obj_aabboxes.append(grasp_obj_aabbox)
                # Store grasp object position, orientation (yaw), dimensions
                object_positions.append(grasp_obj_position)
                object_yaws.append(grasp_obj_yaw)
                objects_dimensions.append(grasp_objs_dimensions[idx])
                break

    # Now add a random orientation to the tabular obstacle and move all the grasp objects placed on it accordingly
    tab_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
    tab_orientation = Rotation.from_euler("xyz", [torch.pi/2,0,tab_yaw], degrees=False).as_quat()[np.array([3,0,1,2])] 
    obstacles[tab_index].set_world_pose(orientation=torch.tensor(tab_orientation,dtype=torch.float, device=device))

    for idx, _ in enumerate(grasp_objs):
        object_yaws[idx] += tab_yaw # Add orientation that was just added to tabular obstacle
        if (object_yaws[idx] < -np.pi): object_yaws[idx] + 2*np.pi, # ensure within -pi to pi
        if (object_yaws[idx] >  np.pi): object_yaws[idx] - 2*np.pi, # ensure within -pi to pi
        # modify x-y positions of grasp objects accordingly
        curr_rel_x, curr_rel_y = object_positions[idx][0] - tab_position[0], object_positions[idx][1] - tab_position[1] # Get relative co-ords
        modify_x, modify_y = curr_rel_x*np.cos(tab_yaw) - curr_rel_y*np.sin(tab_yaw), curr_rel_x*np.sin(tab_yaw) + curr_rel_y*np.cos(tab_yaw)
        new_x, new_y = modify_x + tab_position[0], modify_y + tab_position[1]
        object_positions[idx] = [new_x.item(), new_y.item(), object_positions[idx][2].item()] # new x and y but z is unchanged
        obj_orientation = Rotation.from_euler("xyz", [-torch.pi/2,0,object_yaws[idx]], degrees=False).as_quat()[np.array([3,0,1,2])]
        
        grasp_objs[idx].set_world_pose(position=torch.tensor(object_positions[idx],dtype=torch.float,device=device),
                                       orientation=torch.tensor(obj_orientation,dtype=torch.float,device=device))
    
    # Store tabular obstacle position, orientation, dimensions and AABBox
    object_positions.append(tab_position)
    object_yaws.append(tab_yaw)
    objects_dimensions.append(obstacles_dimensions[tab_index])
    self._scene._bbox_cache.Clear()
    obst_aabboxes.append(self._scene.compute_object_AABB(obstacles[tab_index].name))

    # Create tabular object oriented BBox
    if return_obstacles:
        bbox_tf = np.zeros((3,3))
        bbox_tf[:2,:2] = np.array([[np.cos(object_yaws[-1]), -np.sin(object_yaws[-1])],[np.sin(object_yaws[-1]), np.cos(object_yaws[-1])]])
        bbox_tf[:,-1] = np.array([object_positions[-1][0], object_positions[-1][1], 1.0]) # x,y,1
        min_xy_vertex = np.array([[objects_dimensions[-1][0,0],objects_dimensions[-1][0,1],1.0]]).T
        max_xy_vertex = np.array([[objects_dimensions[-1][1,0],objects_dimensions[-1][1,1],1.0]]).T
        new_min_xy_vertex = (bbox_tf @ min_xy_vertex)[0:2].T.squeeze()
        new_max_xy_vertex = (bbox_tf @ max_xy_vertex)[0:2].T.squeeze()
        z_top_to_ground = object_positions[-1][2] + objects_dimensions[-1][1,2] # z position plus distance to object top
        # Oriented bbox: x0, y0, x1, y1, z_to_ground, yaw
        obst_oriented_bbox = torch.tensor([ new_min_xy_vertex[0], new_min_xy_vertex[1],
                                       new_max_xy_vertex[0], new_max_xy_vertex[1],
                                            z_top_to_ground,     object_yaws[-1], ] ,dtype=torch.float,device=device)    
    
    # Now we need to place all the other obstacles (without overlaps):
    for idx, _ in enumerate(obstacles):
        if (idx == tab_index): continue # Skip this since we have already placed tabular obstacle

        obst_xyz_size = obstacles_dimensions[idx][1] - obstacles_dimensions[idx][0]
        obst_z_to_ground = - obstacles_dimensions[idx][0,2]
        
        while(1): # Be careful about infinite loops!
            # Place obstacle at random position and orientation on the ground plane
            # polar co-ords
            obst_r = np.random.uniform(robot_radius+np.max(obst_xyz_size[0:2]),world_xy_radius) # taking max xy size margin from robot
            if scene_type == 'complex':
                # Place other obstacle close to the table. Keep its phi within 60 degrees of table's phi
                obst_phi = np.random.uniform(tab_phi-np.pi/3.0,tab_phi+np.pi/3.0)
            else:
                obst_phi = np.random.uniform(-np.pi,np.pi)

            obst_x, obst_y = obst_r*np.cos(obst_phi), obst_r*np.sin(obst_phi)
            obst_position = [obst_x.item(),obst_y.item(),obst_z_to_ground.item()]
            obst_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
            obst_orientation = Rotation.from_euler("xyz", [torch.pi/2,0,obst_yaw], degrees=False).as_quat()[np.array([3,0,1,2])]
            obstacles[idx].set_world_pose(position=torch.tensor(obst_position,dtype=torch.float,device=device),
                                          orientation=torch.tensor(obst_orientation,dtype=torch.float,device=device))
    # Store tabular obstacle position, orientation, dimensions and AABBox
            # compute new AxisAligned bbox
            self._scene._bbox_cache.Clear()
            obst_aabbox = self._scene.compute_object_AABB(obstacles[idx].name)
            # Check for overlap with all existing grasp objects
            overlap = False
            for other_aabbox in obst_aabboxes: # loop over existing AAbboxes
                obst_range = Gf.Range3d(Gf.Vec3d(obst_aabbox[0,0],obst_aabbox[0,1],obst_aabbox[0,2]),Gf.Vec3d(obst_aabbox[1,0],obst_aabbox[1,1],obst_aabbox[1,2]))
                other_obst_range = Gf.Range3d(Gf.Vec3d(other_aabbox[0,0],other_aabbox[0,1],other_aabbox[0,2]),Gf.Vec3d(other_aabbox[1,0],other_aabbox[1,1],other_aabbox[1,2]))
                intersec = Gf.Range3d.GetIntersection(obst_range, other_obst_range)
                if (not intersec.IsEmpty()):
                    overlap = True # Failed. Try another pose
                    break
            if (overlap):
                continue # Failed. Try another pose
            else:
                # Success. Add this valid AAbbox to the list
                obst_aabboxes.append(obst_aabbox)
                # Store obstacle position, orientation (yaw) and dimensions
                object_positions.append(obst_position)
                object_yaws.append(obst_yaw)
                objects_dimensions.append(obstacles_dimensions[idx])
                break

    # All objects placed in the scene!
    # Pick one object to be the grasp object and compute its grasp:
    random_goal_obj = False
    if random_goal_obj:
        if scene_type == 'complex':
            # Don't pick the big objects as the goal object
            goal_obj_index = np.random.randint(len(grasp_objs)-2)
        else:
            goal_obj_index = np.random.randint(len(grasp_objs))
    else:
        # Optional: try to select the most occluded object as the goal object
        # simply choose the one furthest away from origin
        positions_to_consider = np.array(object_positions)[:len(grasp_objs)-2]
        positions_to_consider = positions_to_consider[:,:2] # only need x-y
        goal_obj_index = np.argmax(np.linalg.norm(positions_to_consider, axis=1))

    # For now, generating only top grasps: no roll, pitch 90, same yaw as object
    goal_roll = 0.0 # np.random.uniform(-np.pi,np.pi)
    goal_pitch = np.pi/2.0 # np.random.uniform(0,np.pi/2.0)
    goal_yaw = object_yaws[goal_obj_index]
    goal_position = np.array(object_positions[goal_obj_index])
    goal_position[2] = grasp_obj_aabboxes[goal_obj_index][0,2] # bottom of object z value
    goal_position[:2] = goal_position[:2] + np.random.uniform(-0.04, 0.04, 2) # Add (random) x and y offset to object center 

    goal_orientation = Rotation.from_euler("xyz", [goal_roll,goal_pitch,goal_yaw], degrees=False).as_quat()[np.array([3,0,1,2])]
    goal_pose = torch.hstack(( torch.tensor(goal_position,dtype=torch.float,device=device),
                        torch.tensor(goal_orientation,dtype=torch.float,device=device)))

    # adjust aabbox to uncertainty of +- 4 cm in xy directions
    grasp_aabbox = self._scene.compute_object_AABB(grasp_objs[goal_obj_index].name)
    grasp_aabbox[0][:2] -= 0.04
    grasp_aabbox[1][:2] += 0.04
    grasp_aabbox[1, 2] += 0.1


    # Remove the goal object from obj_positions and yaws list (for computing oriented bboxes)
    del object_positions[goal_obj_index], object_yaws[goal_obj_index]
    # Remove from objects_dimensions list (for computing collision bboxes)
    del objects_dimensions[goal_obj_index]
    
    # Compute oriented bounding boxes for all remaining objects
    for idx in range(len(object_positions)):
        bbox_tf = np.zeros((3,3))
        bbox_tf[:2,:2] = np.array([[np.cos(object_yaws[idx]), -np.sin(object_yaws[idx])],[np.sin(object_yaws[idx]), np.cos(object_yaws[idx])]])
        bbox_tf[:,-1] = np.array([object_positions[idx][0], object_positions[idx][1], 1.0]) # x,y,1
        min_xy_vertex = np.array([[objects_dimensions[idx][0,0],objects_dimensions[idx][0,1],1.0]]).T
        max_xy_vertex = np.array([[objects_dimensions[idx][1,0],objects_dimensions[idx][1,1],1.0]]).T
        new_min_xy_vertex = (bbox_tf @ min_xy_vertex)[0:2].T.squeeze()
        new_max_xy_vertex = (bbox_tf @ max_xy_vertex)[0:2].T.squeeze()
        z_top_to_ground = object_positions[idx][2] + objects_dimensions[idx][1,2] # z position plus distance to object top
        # Oriented bbox: x0, y0, x1, y1, z_to_ground, yaw
        oriented_bbox = torch.tensor([ new_min_xy_vertex[0], new_min_xy_vertex[1],
                                       new_max_xy_vertex[0], new_max_xy_vertex[1],
                                            z_top_to_ground,     object_yaws[idx], ] ,dtype=torch.float,device=device)
        if idx == 0:
            object_oriented_bboxes = oriented_bbox
        else:
            object_oriented_bboxes = torch.vstack(( object_oriented_bboxes, oriented_bbox ))

        # Compute collision 2D bboxes (using shapely) of obstacle objects
        if idx >= len(grasp_objs) - 1:
            # expand collision box to avoid collisions with robot
            min_xy_vertex -= obst_collision_expansion
            max_xy_vertex += obst_collision_expansion
            # transform and scale based on occ map cell size and radius
            min_xy_vertex /= occ_map_cell_size
            max_xy_vertex /= occ_map_cell_size
            bbox_tf[:,-1] += occ_map_radius
            bbox_tf[:,-1] /= occ_map_cell_size
            # create shapely box
            collision_box = geometry.box(min_xy_vertex[0], min_xy_vertex[1],
                                        max_xy_vertex[0], max_xy_vertex[1]) # minx, miny, maxx, maxy, ccw=True
            # transform based on the pose we have just set
            matrix = [bbox_tf[0,0], bbox_tf[0,1], bbox_tf[1,0], bbox_tf[1,1], bbox_tf[0,2], bbox_tf[1,2]]
            collision_box = shapely.affinity.affine_transform(collision_box,matrix)
            obst_collision_2D_boxes.append(collision_box)


    if return_obstacles:
        if return_goal_aabbox:
            return grasp_objs[goal_obj_index], goal_pose, object_oriented_bboxes, obst_oriented_bbox, grasp_aabbox, obst_collision_2D_boxes
        else:
            return grasp_objs[goal_obj_index], goal_pose, object_oriented_bboxes, obst_oriented_bbox, obst_collision_2D_boxes
    else:
        if return_goal_aabbox:
            return grasp_objs[goal_obj_index], goal_pose, object_oriented_bboxes, grasp_aabbox, obst_collision_2D_boxes
        else:
            return grasp_objs[goal_obj_index], goal_pose, object_oriented_bboxes, obst_collision_2D_boxes
