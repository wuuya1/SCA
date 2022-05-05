import os
import time
import json
import math
import random
import pandas as pd
import numpy as np
from mamp.agents.agent import Agent
from mamp.agents.obstacle import Obstacle
from mamp.policies.orca3dPolicy import ORCA3DPolicy
from mamp.configs.config import DT
from mamp.envs.mampenv import MACAEnv
from mamp.util import mod2pi


def set_circle_pos(agent_num):
    """
    exp4: The rad is 18.0 for evaluation.
    """
    center = (0.0, 0.0)
    rad = 18.0
    k = 0
    agent_origin_pos = []
    agent_pos = []
    agent_goal = []
    for j in range(agent_num):
        agent_pos.append([center[0] + round(rad * np.cos(2 * j * np.pi / agent_num + k * np.pi / 4), 2),
                          center[1] + round(rad * np.sin(2 * j * np.pi / agent_num + k * np.pi / 4), 2),
                          10.0,
                          round(mod2pi(2 * j * np.pi / agent_num + k * np.pi / 4 + np.pi), 5), 0, 0])
        agent_origin_pos.append((agent_pos[j][0], agent_pos[j][1]))

    for j in range(agent_num):
        agent_goal.append(agent_pos[(j + int(agent_num / 2)) % agent_num][:3]
                          + [round(mod2pi(2 * j * np.pi / agent_num + k * np.pi / 4 + np.pi), 5), 0.0, 0.0])
    return agent_pos, agent_goal, agent_origin_pos


def set_sphere(num_N):
    """
    exp4: The rad is 25.0 and z_value is 30.0 for evaluation.
    """
    agent_origin_pos = []
    agent_pos = []
    agent_goal = []
    rad = 25.0
    z_value = 30.0
    param_phi = (math.sqrt(5.0) - 1.0) / 2.0
    for n in range(1, num_N + 1):
        z_n = (2 * n - 1) / num_N - 1
        x_n = math.sqrt(1 - z_n ** 2) * math.cos(2 * math.pi * n * param_phi)
        y_n = math.sqrt(1 - z_n ** 2) * math.sin(2 * math.pi * n * param_phi)
        pos = np.array([rad * x_n, rad * y_n, rad * z_n, 0.0, 0.0, 0.0])
        agent_pos.append(pos)
        agent_goal.append(-pos)
    for i in range(len(agent_pos)):
        agent_pos[i][2] = agent_pos[i][2] + z_value
        agent_goal[i][2] = agent_goal[i][2] + z_value

    return agent_pos, agent_goal, agent_origin_pos


def set_random_pos(agent_num):
    """
    exp4: The r is 15.0, z_value is 30.0.
    If collision occurs in the start stage, run this program again.
    """
    agent_origin_pos = []
    agent_pos = []
    agent_goal = []
    z_value = 30.0
    r = 15.0
    for n in range(agent_num):
        pos = np.array([random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r), 0.0, 0.0, 0.0])
        agent_pos.append(pos)
    for n in range(agent_num):
        pos = np.array([random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r), 0.0, 0.0, 0.0])
        agent_goal.append(pos)

    for i in range(len(agent_pos)):
        agent_pos[i][2] = agent_pos[i][2] + z_value
        agent_goal[i][2] = agent_goal[i][2] + z_value
    return agent_pos, agent_goal, agent_origin_pos


def build_agents():
    # build agents
    drone_num = 100
    init_vel = [0.0, 0.0, 0.0]
    radius = 0.5
    pref_speed = 1.0
    pos, goal, DroneOriginPos = set_circle_pos(drone_num)
    # pos, goal, DroneOriginPos = set_sphere(drone_num)
    # pos, goal, DroneOriginPos = set_random_pos(drone_num)
    agents = []
    for i in range(len(pos)):
        agents.append(Agent(start_pos=pos[i], goal_pos=goal[i],
                            vel=init_vel, radius=radius,
                            pref_speed=pref_speed, policy=ORCA3DPolicy,
                            id=i, dt=DT))
    return agents


def build_obstacles():
    ob_list = [[0.0, 0.0, 10.0]]
    obstacles = []
    # obstacles.append(Obstacle(pos=ob_list[0], shape_dict={'shape': "cube", 'feature': (2.0, 2.0, 2.0)}, id=0))
    return obstacles


if __name__ == "__main__":
    total_time = 10000
    step = 0

    # Build agents and obstacles.
    agents = build_agents()
    obstacles = build_obstacles()

    env = MACAEnv()
    env.set_agents(agents, obstacles=obstacles)

    # Test for ORCA.
    agents_num = len(agents)
    cost_time = 0.0
    start_time = time.time()
    while step * DT < total_time:
        print(step, '')

        actions = {}
        which_agents_done = env.step(actions)

        # Is arrived.
        if which_agents_done:
            print("All agents finished!", step)
            end_time = time.time()
            cost_time = end_time - start_time
            print('cost time', cost_time)
            break
        step += 1

    log_save_dir = os.path.dirname(os.path.realpath(__file__)) + '/../visualization/orca3d/log/'
    os.makedirs(log_save_dir, exist_ok=True)
    # trajectory
    writer = pd.ExcelWriter(log_save_dir + '/trajs.xlsx')
    for a in agents:
        a.history_info.to_excel(writer, sheet_name='agent' + str(a.id))
    writer.save()

    # scenario information
    info_dict_to_visualize = {
        'all_agent_info': [],
        'all_obstacle': [],
        'all_compute_time': 0.0,
        'all_straight_distance': 0.0,
        'all_distance': 0.0,
        'successful_num': 0,
        'all_desire_step_num': 0,
        'all_step_num': 0,
        'SuccessRate': 0.0,
        'ExtraTime': 0.0,
        'ExtraDistance': 0.0,
        'AverageSpeed': 0.0,
        'AverageCost': 0.0
    }
    all_straight_dist = 0.0
    all_agent_total_time = 0.0
    all_agent_total_dist = 0.0
    num_of_success = 0
    all_desire_step_num = 0
    all_step_num = 0

    SuccessRate = 0.0
    ExtraTime = 0.0
    ExtraDistance = 0.0
    AverageSpeed = 0.0
    AverageCost = 0.0

    for agent in agents:
        agent_info_dict = {'id': agent.id, 'gp': agent.group, 'radius': agent.radius,
                           'goal_pos': agent.goal_global_frame.tolist()}
        info_dict_to_visualize['all_agent_info'].append(agent_info_dict)
        if not agent.is_collision and not agent.is_out_of_max_time:
            num_of_success += 1
            all_agent_total_time += agent.total_time
            all_straight_dist += agent.straight_path_length
            all_agent_total_dist += agent.total_dist
            all_desire_step_num += agent.desire_steps
            all_step_num += agent.step_num

    info_dict_to_visualize['all_compute_time'] = all_agent_total_time
    info_dict_to_visualize['all_straight_distance'] = all_straight_dist
    info_dict_to_visualize['all_distance'] = all_agent_total_dist
    info_dict_to_visualize['successful_num'] = num_of_success
    info_dict_to_visualize['all_desire_step_num'] = all_desire_step_num
    info_dict_to_visualize['all_step_num'] = all_step_num

    info_dict_to_visualize['SuccessRate'] = num_of_success / agents_num
    info_dict_to_visualize['ExtraTime'] = ((all_step_num - all_desire_step_num) * DT) / num_of_success
    info_dict_to_visualize['ExtraDistance'] = (all_agent_total_dist - all_straight_dist) / num_of_success
    info_dict_to_visualize['AverageSpeed'] = all_agent_total_dist/all_step_num/DT
    info_dict_to_visualize['AverageCost'] = 1000*all_agent_total_time / all_step_num

    for obstacle in obstacles:
        obstacle_info_dict = {'position': obstacle.pos, 'shape': obstacle.shape, 'feature': obstacle.feature}
        info_dict_to_visualize['all_obstacle'].append(obstacle_info_dict)

    info_str = json.dumps(info_dict_to_visualize, indent=4)
    with open(log_save_dir + '/env_cfg.json', 'w') as json_file:
        json_file.write(info_str)
    json_file.close()
    step = 0

