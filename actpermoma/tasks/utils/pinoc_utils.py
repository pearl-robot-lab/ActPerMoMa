from __future__ import print_function
import numpy as np
import os
import pinocchio as pin
from scipy.spatial.transform import Rotation


def get_se3_err(pos_first, quat_first, pos_second, quat_second):
    # Retruns 6 dimensional log.SE3 error between two poses expressed as position and quaternion rotation
    
    rot_first = Rotation.from_quat(np.array([quat_first[1],quat_first[2],quat_first[3],quat_first[0]])).as_matrix() # Quaternion in scalar last format!!!
    rot_second = Rotation.from_quat(np.array([quat_second[1],quat_second[2],quat_second[3],quat_second[0]])).as_matrix() # Quaternion in scalar last format!!!
    
    oMfirst = pin.SE3(rot_first, pos_first)
    oMsecond = pin.SE3(rot_second, pos_second)
    firstMsecond = oMfirst.actInv(oMsecond)
    
    return pin.log(firstMsecond).vector # log gives us a spatial vector (exp co-ords)


class PinTiagoIKSolver(object):
    def __init__(
        self,
        urdf_name: str = "tiago_dual_holobase.urdf",
        move_group: str = "arm_right", # Can be 'arm_right' or 'arm_left'
        include_torso: bool = False, # Use torso in th IK solution
        include_base: bool = False, # Use base in th IK solution
        include_base_rotation: bool = False, # Use base rotation in the IK solution, only relevant/considered if include_base is False
        include_head: bool = False, # Use head in th IK solution
        max_rot_vel: float = 1.0472
    ) -> None:
        # Settings
        self.damp = 1e-10 # Damping co-efficient for linalg solve (to avoid singularities)
        self._include_torso = include_torso
        self._include_base = include_base
        self._include_base_rotation = include_base_rotation
        self._include_head = include_head
        self.max_rot_vel = max_rot_vel # Maximum rotational velocity of all joints
        
        ## Load urdf
        urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../urdf/" + urdf_name
        self.model = pin.buildModelFromUrdf(urdf_file)
        # Choose joints
        name_end_effector = "gripper_"+move_group[4:]+"_grasping_frame"

        jointsOfInterest = [move_group+'_1_joint', move_group+'_2_joint',
                            move_group+'_3_joint', move_group+'_4_joint', move_group+'_5_joint',
                            move_group+'_6_joint', move_group+'_7_joint']
        if self._include_torso:
            # Add torso joint
            jointsOfInterest = ['torso_lift_joint'] + jointsOfInterest
        if self._include_base:
            # Add base joints
            jointsOfInterest = ['X','Y','R'] + jointsOfInterest # 10 DOF with holo base joints included (11 with torso)
        elif self._include_base_rotation:
            # Add base rotation
            jointsOfInterest = ['R'] + jointsOfInterest
        if self._include_head:
            # Add head joints
            jointsOfInterest = jointsOfInterest + ['head_1_joint','head_2_joint'] # 13 DOF with holo base joints & torso included

        remove_ids = list()
        for jnt in jointsOfInterest:
            if self.model.existJointName(jnt):
                remove_ids.append(self.model.getJointId(jnt))
            else:
                print('[IK WARNING]: joint ' + str(jnt) + ' does not belong to the model!')
        jointIdsToExclude = np.delete(np.arange(0,self.model.njoints), remove_ids)
        # Lock extra joints except joint 0 (root)
        reference_configuration=pin.neutral(self.model)
        if not self._include_torso:
            reference_configuration[26] = 0.25 # lock torso_lift_joint at 0.25
        self.model = pin.buildReducedModel(self.model, jointIdsToExclude[1:].tolist(), reference_configuration=reference_configuration)
        assert (len(self.model.joints)==(len(jointsOfInterest)+1)), "[IK Error]: Joints != nDoFs"
        self.model_data = self.model.createData()
        # Define Joint-Limits
        self.joint_pos_min = np.array([-1.1780972451, -1.1780972451, -0.785398163397, -0.392699081699, -2.09439510239, -1.41371669412, -2.09439510239])
        self.joint_pos_max = np.array([+1.57079632679, +1.57079632679, +3.92699081699, +2.35619449019, +2.09439510239, +1.41371669412, +2.09439510239])
        if self._include_torso:
            self.joint_pos_min = np.hstack((np.array([0.0]), self.joint_pos_min))
            self.joint_pos_max = np.hstack((np.array([0.35]), self.joint_pos_max))
        if self._include_base:
            self.joint_pos_min = np.hstack((np.array([-100.0, -100.0, -1.0, -1.0]), self.joint_pos_min))
            self.joint_pos_max = np.hstack((np.array([+100.0, +100.0, +1.0, +1.0]),self.joint_pos_max))
        elif self._include_base_rotation:
            self.joint_pos_min = np.hstack((np.array([-1.0, -1.0]), self.joint_pos_min))
            self.joint_pos_max = np.hstack((np.array([+1.0, +1.0]), self.joint_pos_max))
        if self._include_head:
            self.joint_pos_min = np.hstack((self.joint_pos_min, np.array([-1.24, -0.98])))
            self.joint_pos_max = np.hstack((self.joint_pos_max, np.array([+1.24, +0.72])))
            
        self.joint_pos_mid = (self.joint_pos_max + self.joint_pos_min)/2.0
        # Get End Effector Frame ID
        self.name_end_effector = name_end_effector
        self.id_EE = self.model.getFrameId(name_end_effector)

    def solve_fk_tiago(self, curr_joints, frame=None):
        """Get current Cartesian Pose of specified frame or end-effector if not specified

        @param curr_joints: list or ndarray of joint states with shape (num_joints,), joints referring to the ones set in __init__ as joints of interest
        @param frame: str name of frame to get pose of
        @returns pos
                 np.array(w, x, y, z) rotation as quat
        """
        if frame is None:
            frame_id = self.id_EE
        else:
            frame_id = self.model.getFrameId(frame)

        pin.framesForwardKinematics(self.model,self.model_data, curr_joints)
        oMf = self.model_data.oMf[frame_id]
        pos = oMf.translation
        quat = pin.Quaternion(oMf.rotation)

        return pos, np.array([quat.w, quat.x, quat.y, quat.z])

    def solve_ik_pos_tiago(self, des_pos, des_quat, curr_joints=None, frame=None, n_trials=7, dt=0.1, pos_threshold=0.05, angle_threshold=15.*np.pi/180, verbose=False, debug=False):
        """Get IK joint positions for desired pose of frame or end-effector if not specified

        @param des_pos, des_quat: desired pose, Quaternion in scalar first format!!!
        @param curr_joints: list or ndarray of joint states with shape (num_joints,), joints referring to the ones set in __init__ as joints of interest
        @param frame: str name of frame to compute IK for
        @returns success: boolean
                 best_q: ndarray of joint states
        """
        damp = 1e-10
        success = False
        
        if des_quat is not None:
            # quaternion to rot matrix
            des_rot = Rotation.from_quat(np.array([des_quat[1],des_quat[2],des_quat[3],des_quat[0]])).as_matrix() # Quaternion in scalar last format!!!
            oMdes = pin.SE3(des_rot, des_pos)
        else:
            # 3D position error only
            des_rot = None

        if curr_joints is None:
            curr_joints = np.random.uniform(self.joint_pos_min, self.joint_pos_max)

        if frame is None:
            frame_id = self.id_EE
        else:
            frame_id = self.model.getFrameId(frame)

        q = curr_joints
        q_list = [] # list of joint states over iterations

        for n in range(n_trials):
            for i in range(800): # was 800
                pin.framesForwardKinematics(self.model,self.model_data, q)
                oMf = self.model_data.oMf[frame_id]
                if des_rot is None:
                    oMdes = pin.SE3(oMf.rotation, des_pos) # Set rotation equal to current rotation to exclude this error
                dMf = oMdes.actInv(oMf)
                err = pin.log(dMf).vector
                if (np.linalg.norm(err[0:3]) < pos_threshold) and (np.linalg.norm(err[3:6]) < angle_threshold):
                    success = True
                    break
                J = pin.computeFrameJacobian(self.model,self.model_data,q,frame_id)
                if des_rot is None:
                    J = J[:3,:] # Only pos errors
                    err = err[:3]
                v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
                q = pin.integrate(self.model,q,v*dt)
                # Clip q to within joint limits
                q = np.clip(q, self.joint_pos_min, self.joint_pos_max)

                # debug
                q_list.append(q)
                if verbose:
                    if not i % 100:
                        print('Trial %d: iter %d: error = %s' % (n+1, i, err.T))
                    i += 1
            if success:
                best_q = np.array(q)
                break
            else:
                # Save current solution
                best_q = np.array(q)
                # Reset q to random configuration
                q = np.random.uniform(self.joint_pos_min, self.joint_pos_max)
        if verbose:
            if success:
                print("[[[[IK: Convergence achieved!]]]")
            else:
                print("[Warning: the IK iterative algorithm has not reached convergence to the desired precision]")
        
        if debug:
            return success, best_q, q_list
        else:
            return success, best_q
