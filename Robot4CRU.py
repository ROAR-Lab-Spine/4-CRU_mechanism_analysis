from __future__ import print_function
import numpy as np
import numpy.matlib
import numpy.linalg as la
import math
from sympy import * # symbolic calculation for IK
import transformations as tfs
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

class Robot4CRU(object):
    """class of Robot_4CRU"""
    def __init__(self):
        super(Robot4CRU, self).__init__()
        self.set_geometric_params([0, 0, 1, 4]) # A min alpha, beta = 45 deg, long ee diag length, long base diag length
        #self.set_geometric_params([4, 0, 1, 4]) # B max alpha, beta = 45 deg, long ee diag length, long base diag length
        self.operation_mode = "H1" # H1 = Schoenfiles, H3 = Additional, H2 = Reverse Schoenfiles
        
    def set_geometric_params(self, new_geometric_indcs):
        # End-effector joint axis distribution angles (5 cases)
        self.alphas = np.sort(np.array([np.pi/4.0, np.pi/4.0 + np.pi/8.0, np.pi/2.0, np.pi/2.0 + np.pi/8.0, np.pi/2.0 + np.pi/4.0])/2.0)
        # Base joint axis distribution angles (2 cases)
        self.betas = np.sort(np.array([np.pi/4.0, np.pi/4.0 + np.pi/8.0]))
        # end effector diagonal lengths (2 cases) unit in mm
        self.eeff_diag_lengths = np.sort(np.array([6.25/np.cos(np.pi/8)-2.0, 6.25/np.cos(np.pi/8)]))*25.4
        # base diagonal length (5 cases) unit in mm
        self.base_diag_lengths = np.sort(np.array([self.eeff_diag_lengths[0],
            self.eeff_diag_lengths[1],
            self.eeff_diag_lengths[0]*np.cos(self.alphas[1])/np.cos(self.betas[0]), 
            self.eeff_diag_lengths[1]*np.cos(self.alphas[0])/np.cos(self.betas[0]),
            self.eeff_diag_lengths[1]*np.cos(self.alphas[1])/np.cos(self.betas[0])]))
        # length of the UU couple (from CAD) unit in mm
        self.r = (1.75*2 + 2.54*2)*10.0
        self.joint_pos_range = np.array([0, 300.00])
        self.joint_pos = np.full(4, self.joint_pos_range[1]/2.0) # home position
        #self.h_offset = 3.0*25.4 # (approx.) TODO: update from real CAD
        self.h_offset = 0 # not offset in simulation

        self.geometric_indcs = new_geometric_indcs
        self.a = self.base_diag_lengths[self.geometric_indcs[3]]/2.0*np.cos(self.betas[self.geometric_indcs[1]])
        self.b = self.base_diag_lengths[self.geometric_indcs[3]]/2.0*np.sin(self.betas[self.geometric_indcs[1]])
        self.c = self.eeff_diag_lengths[self.geometric_indcs[2]]/2.0*np.cos(self.alphas[self.geometric_indcs[0]])
        self.d = self.eeff_diag_lengths[self.geometric_indcs[2]]/2.0*np.sin(self.alphas[self.geometric_indcs[0]])
        self.update_geometric_cond()
        
    def update_geometric_cond(self):
        curr_a = round(self.a, 3) # magnitude only
        curr_b = round(self.b, 3)
        curr_c = round(self.c, 3)
        curr_d = round(self.d, 3)
        if (curr_a < curr_c and curr_b > curr_d) or (curr_a > curr_c and curr_b < curr_d):
            self.geometric_cond = "Generic"
        elif curr_a == curr_c and curr_b != curr_d:
            self.geometric_cond = "A"
        elif curr_a != curr_c and curr_b == curr_d:
            self.geometric_cond = "B"
        elif curr_a == curr_c and curr_b == curr_d:
            self.geometric_cond = "C"
        else:
            self.geometric_cond = "Other"
            print("Additional Mode is not achievable!")
        print("Geometric Condition: ", self.geometric_cond)
        print("[a, b, c, d] = ", [self.a, self.b, self.c, self.d], " mm")
        print("r = ", self.r, " mm")
        
    def inverse_kinematics(self, des_pose_4dof):
        reals, discriminants, has_solution = self.check_ik_feasible(des_pose_4dof)
        all_joint_pos_sol = np.full((4, 2), self.joint_pos_range[1]/2.0) # home position as default value
        all_swivel_angles = np.zeros((4, 2)) # indices [joint_no, U1/U2]
        current_joint_pos = self.joint_pos
        selected_joint_pos_sol = current_joint_pos
        if has_solution:
            for i in range(4):
                for j in range(2):
                    if j == 0:
#                         all_joint_pos_sol[i,j] = reals[i] - np.sqrt(discriminants[i]) - self.h_offset
                        all_joint_pos_sol[i,j] = reals[i] - np.sqrt(discriminants[i]) 
                    elif j == 1:
#                         all_joint_pos_sol[i,j] = reals[i] + np.sqrt(discriminants[i]) - self.h_offset
                        all_joint_pos_sol[i,j] = reals[i] + np.sqrt(discriminants[i]) 
#                     swivel_angle_U1, swivel_angle_U2 = self.calc_swivel_angle(des_pose_4dof, all_joint_pos_sol[i,1], i)
#                     all_swivel_angles[i,0] = swivel_angle_U1
#                     all_swivel_angles[i,1] = swivel_angle_U2
                # choose based on the closest joint position
                #joint_pos_diff_j_index = np.argmin(np.abs(all_joint_pos_sol[i,:] - current_joint_pos[i]))
                #selected_joint_pos_sol[i] = all_joint_pos_sol[i, joint_pos_diff_j_index]
                if self.operation_mode == "H1":
                    # prevent actuation sigularity and self-motion during phi = 0
                    # avoid h1 = h2, h3 = h4 or h1 = h4, h2 = h3 or h1 = h2 = h3 = h4
                    # pairs of the same sign: h1-h3, h2-h4
                    joint_pos_sol_index = np.array([1, 0, 1, 0]) # or [0, 1, 0, 1], 1 = hi, 0 = low
                elif self.operation_mode == "H3":
                    # choose the case h1=h4, h2~=h3
                    joint_pos_sol_index = np.array([0, 0, 1, 0]) # or [0, 0, 0, 1] or [1, 1, 1, 0] or [1, 1, 0, 1]
                selected_joint_pos_sol[i] = all_joint_pos_sol[i, joint_pos_sol_index[i]]
        else:
            # if no solution, return current joint positions
            print("No IK Solution: at least one discriminant is negative: ", discriminants)
        # check the swivel angle limit on both ends of the U-U rod: provide warning on the screen if the joint are out of ranges
        print("IK joint pos solutions: ", selected_joint_pos_sol)
        #print("swivel angles (deg): ", np.rad2deg(all_swivel_angles))
        return selected_joint_pos_sol.tolist()
    
    def solve_inv_kin_traj(self):
        pass
    
    def check_ik_feasible(self, pose_4dof):
        tf_mat_base_to_eeff, X, Y, Z, x_0, x_1, x_2, x_3 = self.convert_pose_4dof_to_tf_mat_and_7var(pose_4dof)
        r = self.r
        reals = np.zeros([4])
        discriminants = np.zeros([4])
        has_solution = True
        
        if self.operation_mode == "H1":
            for i in range(4):
                # convert signs based on joint indices
                if i == 0:
                    a = self.a; b = -self.b; c = self.c; d = -self.d
                elif i == 1:
                    a = self.a; b = self.b; c = self.c; d = self.d
                elif i == 2:
                    a = -self.a; b = self.b; c = -self.c; d = self.d
                elif i == 3:
                    a = -self.a; b = -self.b; c = -self.c; d = -self.d
                    
                discriminants[i] = 4.0*X*c*x_3**2 + 4.0*X*d*x_0*x_3 - 4.0*Y*c*x_0*x_3 - 2.0*Y*d*x_0**2 + 2*Y*d*x_3**2 + 4.0*b*c*x_0*x_3 + 2.0*b*d*x_0**2 - 2.0*b*d*x_3**2 - 4.0*c**2*x_3**2 - 4.0*c*d*x_0*x_3 - X**2 - Y**2 + 2.0*Y*b - b**2 - d**2 + r**2
                reals[i] = Z
                
        elif self.operation_mode == "H3":
            a = self.a
            b = self.b
            c = self.c
            d = self.d
            if self.geometric_cond == "A":
                 for i in range(4):
                        if i == 0 or i == 3:
                            discriminants[i] = -Y**2 - 2.0*Y*b + 2.0*Y*d*x_0**2 - 2.0*Y*d*x_1**2 - b**2 + 2.0*b*d*x_0**2 - 2.0*b*d*x_1**2 + 4.0*d**2*x_0**2*x_1**2 - d**2 + r**2         
                            reals[i] = Z - 2.0*d*x_0*x_1
                        elif i == 1 or i == 2:
                            discriminants[i] = -Y**2 + 2.0*Y*b - 2.0*Y*d*x_0**2 + 2.0*Y*d*x_1**2 - b**2 + 2.0*b*d*x_0**2 - 2.0*b*d*x_1**2 + 4.0*d**2*x_0**2*x_1**2 - d**2 + r**2
                            reals[i] = Z + 2.0*d*x_0*x_1
        else:
            has_solution = False
        
        if np.any(np.sign(discriminants) < 0):
            has_solution = False

        return reals, discriminants, has_solution
    
    def calc_swivel_angle(self, des_pose_4dof, joint_pos, joint_index):
        tf_mat_base_to_eeff, X, Y, Z, x_0, x_1, x_2, x_3 = self.convert_pose_4dof_to_tf_mat_and_7var(des_pose_4dof)
        
        # convert signs based on joint indices
        if joint_index == 0:
            a = self.a; b = -self.b; c = self.c; d = -self.d
        elif joint_index == 1:
            a = self.a; b = self.b; c = self.c; d = self.d
        elif joint_index == 2:
            a = -self.a; b = self.b; c = -self.c; d = self.d
        elif joint_index == 3:
            a = -self.a; b = -self.b; c = -self.c; d = -self.d

        u0_B_to_C = np.array([X - a + c*(x_0**2 + x_1**2 - x_2**2 - x_3**2) + d*(-2.0*x_0*x_3 + 2.0*x_1*x_2),
            Y - b + c*(2.0*x_0*x_3 + 2.0*x_1*x_2) + d*(x_0**2 - x_1**2 + x_2**2 - x_3**2),
            Z + c*(-2.0*x_0*x_2 + 2.0*x_1*x_3) + d*(2.0*x_0*x_1 + 2.0*x_2*x_3) - joint_pos - self.h_offset])
        v0_C_to_ee = -np.array([c*(x_0**2 + x_1**2 - x_2**2 - x_3**2) + d*(-2.0*x_0*x_3 + 2.0*x_1*x_2),
            c*(2.0*x_0*x_3 + 2.0*x_1*x_2) + d*(x_0**2 - x_1**2 + x_2**2 - x_3**2),
            c*(-2.0*x_0*x_2 + 2.0*x_1*x_3) + d*(2.0*x_0*x_1 + 2.0*x_2*x_3)])
        v0_B_to_base = -np.array([a, b, 0])

        swivel_angle_U1 = py_ang(u0_B_to_C, v0_B_to_base)
        swivel_angle_U2 = py_ang(v0_C_to_ee, u0_B_to_C)

        return swivel_angle_U1, swivel_angle_U2
    
    def convert_pose_4dof_to_tf_mat_and_7var(self, pose_4dof):
        if self.operation_mode == "H1":
            t_mat = tfs.translation_matrix((pose_4dof[0], pose_4dof[1], pose_4dof[2]))
            r_mat = tfs.rotation_matrix(pose_4dof[3], (0, 0, 1))
            tf_mat_base_to_eeff = tfs.concatenate_matrices(t_mat, r_mat)
        if self.operation_mode == "H3":
            t_mat = tfs.translation_matrix((0, pose_4dof[1], pose_4dof[2]))
            r_mat = tfs.rotation_matrix(pose_4dof[3], (1, 0, 0))
            tf_mat_base_to_eeff = tfs.concatenate_matrices(t_mat, r_mat)
        else:
            # break down to geometric cases
            pass

        X, Y, Z, x_0, x_1, x_2, x_3 = convert_tf_mat_to_7var(tf_mat_base_to_eeff)

        return tf_mat_base_to_eeff, X, Y, Z, x_0, x_1, x_2, x_3

def convert_tf_mat_to_7var(tf_mat):
    quat = tfs.quaternion_from_matrix(tf_mat)
    xyz = tfs.translation_from_matrix(tf_mat)
    X = xyz[0]; Y = xyz[1]; Z = xyz[2] 
    x_0 = quat[3]; x_1 = quat[0]; x_2 = quat[1]; x_3 = quat[2]
    return X, Y, Z, x_0, x_1, x_2, x_3

def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)