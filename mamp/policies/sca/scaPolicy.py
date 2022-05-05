"""
@ Author: Gang Xu
@ Date: 2022.04.16
@ Details: SCA for multi-agent motion planning
@ Reference: 2020 ICRA Minimal 3D Dubins Path with Bounded Curvature and Pitch Angle
@ Reference: Reciprocal Velocity Obstacles for Real-Time Multi-Agent Navigation
@ Github: https://github.com/MengGuo/RVO_Py_MAS
"""
import time
import numpy as np
from math import sqrt, sin, cos, acos, pi
from mamp.policies.sca import dubinsmaneuver3d
from mamp.configs.config import eps, NEAR_GOAL_THRESHOLD
from mamp.util import sqr, absSq, l3norm, is_parallel, is_intersect, satisfied_constraint
from mamp.util import get_phi, reached, cartesian2spherical


class SCAPolicy(object):
    """ SCAPolicy """

    def __init__(self):
        self.now_goal = None
        self.update_now_goal_dist = 1.0
        self.type = "internal"

    def find_next_action(self, dict_comm, agent, kdTree):
        """
        Function: SCAPolicy compute suitable speed for agents
        """
        start_t = time.time()
        self.get_trajectory(agent)  # Update now_goal.
        v_pref = compute_v_pref(agent)
        vA = agent.vel_global_frame
        if l3norm(vA, [0, 0, 0]) <= 1e-5:
            end_t = time.time()
            delta_t = end_t - start_t
            agent.total_time += delta_t
            vA_post = 0.3 * v_pref
            action = cartesian2spherical(agent, vA_post)
            theta = 0.0
        else:
            pA = agent.pos_global_frame
            RVO_BA_all = []
            agent_rad = agent.radius + 0.05
            computeNeighbors(agent, kdTree)
            for obj in agent.neighbors:
                obj = obj[0]
                pB = obj.pos_global_frame
                if obj.is_at_goal:
                    transl_vB_vA = pA
                else:
                    vB = obj.vel_global_frame
                    transl_vB_vA = pA + 0.5 * (vB + vA)  # Use RVO.
                obj_rad = obj.radius + 0.05

                RVO_BA = [transl_vB_vA, pA, pB, obj_rad + agent_rad]
                RVO_BA_all.append(RVO_BA)
            vA_post = intersect(v_pref, RVO_BA_all, agent)
            end_t = time.time()
            delta_t = end_t - start_t
            agent.total_time += delta_t
            action = cartesian2spherical(agent, vA_post)
            theta = acos(min(np.dot(vA, vA_post) / (np.linalg.norm(vA) * np.linalg.norm(vA_post)), 1.0))

        dist = round(l3norm(agent.pos_global_frame, agent.goal_global_frame), 5)
        if theta > agent.max_heading_change:
            print('agent' + str(agent.id), 'Goal distance：', dist, 'Speed:', action[3], 'Dissatisfied Angle', theta)
        else:
            print('agent' + str(agent.id), 'Goal distance:', dist, 'Speed:', round(action[3], 5))
        return action

    def get_trajectory(self, agent):
        if agent.path:
            if self.now_goal is None:  # First
                self.now_goal = np.array(agent.path.pop(), dtype='float64')
            dis = l3norm(agent.pos_global_frame, self.now_goal)
            dis_nowgoal_globalgoal = l3norm(self.now_goal, agent.goal_global_frame)
            dis_nowgoal_globalpos = l3norm(agent.pos_global_frame, agent.goal_global_frame)
            if dis <= self.update_now_goal_dist * agent.radius:  # Free collision.
                if agent.path:
                    self.now_goal = np.array(agent.path.pop(), dtype='float64')
            elif dis_nowgoal_globalgoal >= dis_nowgoal_globalpos:
                if agent.path:
                    self.now_goal = np.array(agent.path.pop(), dtype='float64')
        else:
            self.now_goal = agent.goal_global_frame


def compute_dubins(agent):
    qi = np.hstack((agent.pos_global_frame, agent.heading_global_frame))
    qf = np.hstack((agent.goal_global_frame, agent.goal_heading_frame))
    Rmin, pitchlims = agent.turning_radius, agent.pitchlims
    maneuver = dubinsmaneuver3d.dubinsmaneuver3d(qi, qf, Rmin, pitchlims)
    dubins_path = maneuver.path.copy()
    agent.dubins_sampling_size = maneuver.sampling_size
    desire_length = maneuver.length
    points_num = len(dubins_path)
    path = []
    for i in range(points_num):
        path.append((dubins_path.pop()))
    return path, desire_length, points_num


def computeNeighbors(agent, kdTree):
    if agent.is_collision:
        return

    agent.neighbors.clear()
    rangeSq = agent.neighborDist ** 2
    # Check obstacle neighbors.
    kdTree.computeObstacleNeighbors(agent, rangeSq)
    # Check other agents.
    kdTree.computeAgentNeighbors(agent, rangeSq)


def shunted_strategy(agent, v_list):
    """
    @details: the generalized passsing on the right
    """
    opt_v = [v_list[0]]
    opti_flag = True
    vA = agent.vel_global_frame
    i = 1
    while opti_flag:
        if abs(l3norm(v_list[0], vA) - l3norm(v_list[i], vA)) < 5e-2:
            opt_v.append(v_list[i])
            i += 1
            if i == len(v_list):
                break
        else:
            break
    vA_phi_min = min(opt_v, key=lambda v: get_phi(v))
    vA_phi_max = max(opt_v, key=lambda v: get_phi(v))
    opt_v_best = [vA_phi_min, vA_phi_max]
    phi_min = get_phi(vA_phi_min)
    phi_max = get_phi(vA_phi_max)
    if abs(phi_max - phi_min) <= pi:
        vA_opti = min(opt_v_best, key=lambda v: get_phi(v))
    else:
        vA_opti = max(opt_v_best, key=lambda v: get_phi(v))
    return vA_opti


def compute_without_suitV(agent, RVO_BA_all, unsuit_v):
    tc = []
    for RVO_BA in RVO_BA_all:
        p_0 = RVO_BA[0]
        pA = RVO_BA[1]
        pB = RVO_BA[2]
        combined_radius = RVO_BA[3]
        v_dif = np.array(unsuit_v + pA - p_0)
        pApB = pB - pA
        if is_intersect(pA, pB, combined_radius, v_dif) and satisfied_constraint(agent, unsuit_v):
            discr = sqr(np.dot(v_dif, pApB)) - absSq(v_dif) * (absSq(pApB) - sqr(combined_radius))
            tc_v = (np.dot(v_dif, pApB) - sqrt(discr)) / absSq(v_dif)
            if tc_v < 0:
                tc_v = 0.0
            tc.append(tc_v)
    if len(tc) == 0:
        tc = [0.0]
    return tc


def compute_newV_is_suit(agent, RVO_BA_all, new_v):
    suit = True
    if len(RVO_BA_all) == 0:
        if not satisfied_constraint(agent, new_v):
            suit = False
            return suit

    for RVO_BA in RVO_BA_all:
        p_0 = RVO_BA[0]
        pA = RVO_BA[1]
        pB = RVO_BA[2]
        combined_radius = RVO_BA[3]
        v_dif = np.array(new_v + pA - p_0)  # new_v-0.5*(vA+vB) or new_v-vB
        if is_intersect(pA, pB, combined_radius, v_dif) or not satisfied_constraint(agent, new_v):
            suit = False
            break
    return suit


def intersect(v_pref, RVO_BA_all, agent):
    num_N = 128
    param_phi = (sqrt(5.0) - 1.0) / 2.0
    min_speed = 0.5
    suitable_V = []
    unsuitable_V = []
    for rad in np.arange(min_speed, agent.pref_speed + 0.03, agent.pref_speed - min_speed):
        for n in range(1, num_N + 1):
            z_n = (2 * n - 1) / num_N - 1
            x_n = sqrt(1 - z_n ** 2) * cos(2 * pi * n * param_phi)
            y_n = sqrt(1 - z_n ** 2) * sin(2 * pi * n * param_phi)
            new_v = np.array([rad * x_n, rad * y_n, rad * z_n])
            suit = compute_newV_is_suit(agent, RVO_BA_all, new_v)
            if suit:
                suitable_V.append(new_v)
            else:
                unsuitable_V.append(new_v)
    new_v = v_pref[:]
    suit = compute_newV_is_suit(agent, RVO_BA_all, new_v)
    if suit:
        suitable_V.append(new_v)
    else:
        unsuitable_V.append(new_v)
    # ----------------------
    if suitable_V:
        suitable_V.sort(key=lambda v: l3norm(v, v_pref))  # Sort begin at minimum and end at maximum.
        if len(suitable_V) > 1:
            vA_post = shunted_strategy(agent, suitable_V)
        else:
            vA_post = suitable_V[0]
    else:
        tc_V = dict()
        for unsuit_v in unsuitable_V:
            unsuit_v = np.array(unsuit_v)
            tc_V[tuple(unsuit_v)] = 0
            tc = compute_without_suitV(agent, RVO_BA_all, unsuit_v)
            tc_V[tuple(unsuit_v)] = min(tc) + 1e-5
        WT = 0.2
        vA_post = min(unsuitable_V, key=lambda v: ((WT / tc_V[tuple(v)]) + l3norm(v, v_pref)))
    vA_post = np.array([int(vA_post[0] * eps) / eps, int(vA_post[1] * eps) / eps, int(vA_post[2] * eps) / eps])
    return vA_post


def update_dubins(agent):
    dis = l3norm(agent.pos_global_frame, agent.dubins_now_goal)
    sampling_size = agent.dubins_sampling_size
    if dis < sampling_size * 2:
        if agent.dubins_path:
            agent.dubins_now_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')
        else:
            agent.dubins_now_goal = agent.goal_global_frame


def dubins_path_node_pop(agent):
    agent.dubins_last_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')
    agent.dubins_last_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')
    agent.dubins_last_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')
    agent.dubins_last_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')


def compute_v_pref(agent):
    """
        is_parallel(vA, v_pref): —— Whether to leave Dubins trajectory as collision avoidance.
        dis_goal <= k: —— Regard as obstacles-free when the distance is less than k.
        dis < 6 * sampling_size: —— Follow the current Dubins path when not moving away from the current Dubins path.
        theta >= np.deg2rad(100): —— Avoid the agent moving away from the goal position after update Dubins path.
    """
    dis_goal = l3norm(agent.pos_global_frame, agent.goal_global_frame)
    k = 3.0 * agent.turning_radius
    if not agent.is_use_dubins:  # first
        agent.is_use_dubins = True
        dubins_path, desire_length, points_num = compute_dubins(agent)
        agent.dubins_path = dubins_path
        dubins_path_node_pop(agent)
        agent.dubins_now_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')
        dif_x = agent.dubins_now_goal - agent.pos_global_frame
    else:
        update_dubins(agent)
        vA = np.array(agent.vel_global_frame)
        v_pref = agent.v_pref
        dis = l3norm(agent.pos_global_frame, agent.dubins_now_goal)
        max_size = round(6 * agent.dubins_sampling_size, 5)
        pApG = agent.goal_global_frame - agent.pos_global_frame
        theta = round(acos(min(np.dot(vA, pApG) / (np.linalg.norm(vA) * np.linalg.norm(pApG)), 1.0)), 5)
        deg100 = round(np.deg2rad(100), 5)
        min_dist_ob = round(sqrt(agent.neighbors[0][1]), 5) if agent.neighbors else round(agent.neighborDist)
        p0pA = agent.goal_pos[:3] - agent.initial_pos[:3]
        is_zAxis = abs(np.dot(p0pA, [1.0, 0.0, 0.0])) <= 1e-5 and abs(np.dot(p0pA, [0.0, 1.0, 0.0])) <= 1e-5
        condition_dist = (min_dist_ob >= 2.0 * agent.turning_radius) if is_zAxis else False

        if ((is_parallel(vA, v_pref) or dis_goal <= k) and dis < max_size) or (theta >= deg100) or condition_dist:
            update_dubins(agent)
            if agent.dubins_path:
                dif_x = agent.dubins_now_goal - agent.pos_global_frame
            else:
                dif_x = agent.goal_global_frame - agent.pos_global_frame
        else:
            dubins_path, length, points_num = compute_dubins(agent)
            agent.dubins_path = dubins_path
            dubins_path_node_pop(agent)
            agent.dubins_now_goal = np.array(agent.dubins_path.pop()[:3], dtype='float64')
            dif_x = agent.dubins_now_goal - agent.pos_global_frame

    norm = l3norm(dif_x, [0, 0, 0])
    norm_dif_x = dif_x * agent.pref_speed / norm
    v_pref = np.array(norm_dif_x)
    if reached(agent.goal_global_frame, agent.pos_global_frame, bound=0.2):
        v_pref[0] = 0.0
        v_pref[1] = 0.0
        v_pref[2] = 0.0
    agent.v_pref = v_pref
    v_pref = np.array([int(v_pref[0] * eps) / eps, int(v_pref[1] * eps) / eps, int(v_pref[2] * eps) / eps])
    return v_pref


if __name__ == "__main__":
    pass
