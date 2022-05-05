import copy
import numpy as np
from math import sqrt
from mamp.util import pi_2_pi, l3norm
from mamp.policies.kdTree import KDTree
from mamp.configs.config import NEAR_GOAL_THRESHOLD


class MACAEnv(object):
    def __init__(self):
        self.agents = None
        self.centralized_planner = None
        self.obstacles = []
        self.kdTree = None

    def set_agents(self, agents, obstacles=None):
        self.agents = agents
        self.obstacles = obstacles
        self.kdTree = KDTree(self.agents, self.obstacles)
        self.kdTree.buildObstacleTree()

    def step(self, actions):
        self._take_action(actions)
        which_agents_done = self.is_done(NEAR_GOAL_THRESHOLD)
        return which_agents_done

    def _take_action(self, actions):
        self.kdTree.buildAgentTree()

        num_actions_per_agent = 7  # vx, vy, vz, speed, alpha, beta, gamma
        all_actions = np.zeros((len(self.agents), num_actions_per_agent), dtype=np.float32)

        # Update velocity and position.
        for agent_index, agent in enumerate(self.agents):
            if agent.is_at_goal or agent.is_collision or agent.is_out_of_max_time:
                continue
            other_agents = copy.copy(self.agents)
            other_agents.remove(agent)
            dict_comm = {'other_agents': other_agents, 'obstacles': self.obstacles}
            all_actions[agent_index, :] = agent.policy.find_next_action(dict_comm, agent, self.kdTree)

        for i, agent in enumerate(self.agents):
            update_velocitie(agent, all_actions[i, :])
            if not agent.is_at_goal:
                agent.step_num += 1
            self.check_agent_state(agent)  # Check collision.
        for i, agent in enumerate(self.agents):
            if agent.is_collision:
                print('agent' + str(agent.id) + ': collision')

    def is_done(self, near_goal_threshold):
        for ag in self.agents:
            if l3norm(ag.pos_global_frame, ag.goal_global_frame) <= near_goal_threshold:
                ag.is_at_goal = True
            if ag.is_at_goal or ag.is_collision or ag.is_out_of_max_time:
                ag.is_run_done = True
        is_done_condition = np.array([ag.is_run_done for ag in self.agents])
        check_is_done_condition = np.logical_and.reduce(is_done_condition)
        return check_is_done_condition

    def check_agent_state(self, agent):
        # Check collision.
        for ob in self.obstacles:
            dis_a_ob = l3norm(agent.pos_global_frame, ob.pos_global_frame)
            if dis_a_ob <= (agent.radius + ob.radius):
                agent.is_collision = True
        for ag in self.agents:
            if ag.id == agent.id: continue
            dis_a_agent = l3norm(agent.pos_global_frame, ag.pos_global_frame)
            if dis_a_agent <= (agent.radius + ag.radius):
                if not ag.is_at_goal:
                    ag.is_collision = True
                if not agent.is_at_goal:
                    agent.is_collision = True
        if agent.total_dist > agent.max_run_dist:
            agent.is_out_of_max_time = True
            print(' agent' + str(agent.id) + ' is_out_of_max_time')


def update_velocitie(agent, action):
    selected_speed = action[3]

    selected_alpha_heading = pi_2_pi(agent.heading_global_frame[0] + action[4])
    selected_beta_heading = pi_2_pi(agent.heading_global_frame[1] + action[5])
    selected_gamma_heading = pi_2_pi(agent.heading_global_frame[2] + action[6])

    dx = selected_speed * np.cos(selected_beta_heading) * np.cos(selected_alpha_heading) * agent.dt_nominal
    dy = selected_speed * np.cos(selected_beta_heading) * np.sin(selected_alpha_heading) * agent.dt_nominal
    dz = selected_speed * np.sin(selected_beta_heading) * agent.dt_nominal
    length = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    agent.total_dist += length
    agent.pos_global_frame += np.array([dx, dy, dz])

    agent.heading_global_frame = [selected_alpha_heading, selected_beta_heading, selected_gamma_heading]
    agent.vel_global_frame = np.array(action[:3])
    agent.to_vector()

