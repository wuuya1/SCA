import numpy as np
import pandas as pd
from math import pi
from mamp.util import l3norm, l3normsq, takeSecond, sqr
from mamp.configs.config import DT


class Agent(object):
    def __init__(self, start_pos, goal_pos, vel, radius, pref_speed, policy, id, dt=0.1):
        self.group = 0
        self.policy = policy()

        self.initial_pos = np.array(start_pos, dtype='float64')
        self.goal_pos = np.array(goal_pos, dtype='float64')
        self.pos_global_frame = np.array(start_pos[:3], dtype='float64')
        self.goal_global_frame = np.array(goal_pos[:3], dtype='float64')

        self.heading_global_frame = np.array(start_pos[3:], dtype='float64')
        self.goal_heading_frame = np.array(goal_pos[3:], dtype='float64')
        self.initial_heading = np.array(start_pos[3:], dtype='float64')

        self.vel_global_frame = np.array(vel)
        self.radius = radius
        self.turning_radius = 1.5
        self.id = id
        self.pref_speed = pref_speed
        self.pitchlims = [-pi / 4, pi / 4]
        self.min_heading_change = self.pitchlims[0]
        self.max_heading_change = self.pitchlims[1]

        self.neighbors = []
        self.maxNeighbors = 16
        self.neighborDist = 10.0
        self.timeStep = DT
        self.timeHorizon = 10.0
        self.maxSpeed = 1.0
        self.maxAccel = 1.0
        self.safetyFactor = 7.5
        self.is_parallel_neighbor = []
        self.is_obstacle = False
        self.dt_nominal = DT
        self.candinate_num = 256

        self.path = []
        self.dubins_path = []
        self.travelled_traj_node = []
        self.num_node = 10
        self.desire_points_num = int(0)    # for computing time rate dubins
        self.desire_path_length = None           # for computing distance rate dubins

        self.straight_path_length = l3norm(start_pos[:3], goal_pos[:3])-0.5  # for computing distance rate
        self.desire_steps = int(self.straight_path_length / (pref_speed*DT))  # for computing time rate

        self.dubins_now_goal = None
        self.dubins_last_goal = None
        self.v_pref = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.free_collision_time = int(0)

        self.is_back2start = False
        self.total_time = 0.0
        self.total_dist = 0.0
        self.step_num = 0
        # self.current_step = 0
        # self.current_run_dist = 0.0

        self.history_pos = {}
        self.real_path_length = 0.0             # for computing distance rate
        self.is_use_dubins = False
        self.ref_plane = 'XOY'
        self.is_at_goal = False
        self.is_collision = False
        self.is_out_of_max_time = False
        self.is_run_done = False
        self.max_run_dist = 3.0 * l3norm(start_pos[:3], goal_pos[:3])
        self.ANIMATION_COLUMNS = ['pos_x', 'pos_y', 'pos_z', 'alpha', 'beta', 'gamma', 'vel_x', 'vel_y', 'vel_z',
                                  'gol_x', 'gol_y', 'gol_z', 'radius']
        self.history_info = pd.DataFrame(columns=self.ANIMATION_COLUMNS)

    def insertAgentNeighbor(self, other_agent, rangeSq):
        if self.id != other_agent.id:
            distSq = l3normsq(self.pos_global_frame, other_agent.pos_global_frame)
            if distSq < sqr(self.radius + other_agent.radius) and distSq < rangeSq:     # COLLISION!
                if not self.is_collision:
                    self.is_collision = True
                    self.neighbors.clear()

                if len(self.neighbors) == self.maxNeighbors:
                    self.neighbors.pop()
                self.neighbors.append((other_agent, distSq))
                self.neighbors.sort(key=takeSecond)
                if len(self.neighbors) == self.maxNeighbors:
                    rangeSq = self.neighbors[-1][1]
            elif not self.is_collision and distSq < rangeSq:
                if len(self.neighbors) == self.maxNeighbors:
                    self.neighbors.pop()
                self.neighbors.append((other_agent, distSq))
                self.neighbors.sort(key=takeSecond)
                if len(self.neighbors) == self.maxNeighbors:
                    rangeSq = self.neighbors[-1][1]

    def insertObstacleNeighbor(self, obstacle, rangeSq):
        index = self.id
        ob_index = obstacle.id
        distSq1 = l3normsq(self.pos_global_frame, obstacle.pos_global_frame)
        '''适合半径接近或大于neighborDist的时候'''
        distSq = (l3norm(self.pos_global_frame, obstacle.pos_global_frame)-obstacle.radius)**2
        if distSq1 < sqr(self.radius + obstacle.radius) and distSq < rangeSq:  # COLLISION!
            if not self.is_collision:
                self.is_collision = True
                self.neighbors.clear()

            if len(self.neighbors) == self.maxNeighbors:
                self.neighbors.pop()
            self.neighbors.append((obstacle, distSq))
            self.neighbors.sort(key=takeSecond)
            if len(self.neighbors) == self.maxNeighbors:
                rangeSq = self.neighbors[-1][1]
        elif not self.is_collision and distSq < rangeSq:
            if len(self.neighbors) == self.maxNeighbors:
                self.neighbors.pop()
            self.neighbors.append((obstacle, distSq))
            self.neighbors.sort(key=takeSecond)
            if len(self.neighbors) == self.maxNeighbors:
                rangeSq = self.neighbors[-1][1]

    def to_vector(self):
        """ Convert the agent's attributes to a single global state vector. """
        global_state_dict = {
            'radius': self.radius,
            'pref_speed': self.pref_speed,
            'pos_x': self.pos_global_frame[0],
            'pos_y': self.pos_global_frame[1],
            'pos_z': self.pos_global_frame[2],
            'gol_x': self.goal_global_frame[0],
            'gol_y': self.goal_global_frame[1],
            'gol_z': self.goal_global_frame[2],
            'vel_x': self.vel_global_frame[0],
            'vel_y': self.vel_global_frame[1],
            'vel_z': self.vel_global_frame[2],
            'alpha': self.heading_global_frame[0],
            'beta': self.heading_global_frame[1],
            'gamma': self.heading_global_frame[2],
        }
        global_state = np.array([val for val in global_state_dict.values()])
        animation_columns_dict = {}
        for key in self.ANIMATION_COLUMNS:
            animation_columns_dict.update({key: global_state_dict[key]})
        self.history_info = self.history_info.append([animation_columns_dict], ignore_index=True)