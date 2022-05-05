import json
import os
import glob
import imageio
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi

from vis_util import get_uav_model, draw_agent_3d, draw_sphere

plt_colors = [[0.8500, 0.3250, 0.0980], [0.0, 0.4470, 0.7410], [0.4660, 0.6740, 0.1880],
              [0.4940, 0.1840, 0.5560],
              [0.9290, 0.6940, 0.1250], [0.3010, 0.7450, 0.9330], [0.6350, 0.0780, 0.1840]]


def draw_traj_3d(ax, agents_info, obstacles_info, agents_traj_list, step_num_list, current_step):
    for idx, agent_traj in enumerate(agents_traj_list):
        current_step_temp = current_step
        color_ind = idx % len(plt_colors)
        plt_color = plt_colors[color_ind]

        ag_step_num = step_num_list[idx]
        if current_step >= ag_step_num:
            current_step = ag_step_num - 1
        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        pos_z = agent_traj['pos_z']
        alpha = agent_traj['alpha']
        beta = agent_traj['beta']
        gamma = agent_traj['gamma']
        vel_x = agent_traj['vel_x']
        vel_y = agent_traj['vel_y']
        vel_z = agent_traj['vel_z']
        goal = agents_info[idx]['goal_pos']
        agent_radius = agents_info[idx]['radius']

        plt.plot(pos_x[:current_step], pos_y[:current_step], pos_z[:current_step], color=plt_color, ls='-', linewidth=2)
        pos_global_frame = [pos_x[current_step], pos_y[current_step], pos_z[current_step]]
        vel = [vel_x[current_step], vel_y[current_step], vel_z[current_step]]
        heading_global_frame = [alpha[current_step], beta[current_step], gamma[current_step]]
        current_step = current_step_temp

        # -------Use sphere model-------
        c_map = get_cmap(len(agents_traj_list))
        # draw_agent(ax, pos_global_frame, goal, vel, radius, c_map(idx), idx)

        # -------Use uav_model-------
        my_agent_model = get_uav_model(agent_radius, tall=0.1)

        # -------Use car_model-------
        # my_agent_model = get_car_model(agent_radius, height=0.3)

        draw_agent_3d(ax=ax,
                      pos_global_frame=pos_global_frame,
                      goal=goal,
                      c_map=c_map(idx),
                      heading_global_frame=heading_global_frame,
                      my_agent_model=my_agent_model, a_id=idx)

    for obstacle_dict in obstacles_info:
        if obstacle_dict['shape'] == 'sphere':
            draw_sphere(ax, obstacle_dict)


def set_ax_parameter(ax):
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.6, 1]))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.view_init(32, 45)

    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-5, 20)
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')


def plot_save_one_pic(obstacles_info, agents_info, agents_traj_list, step_num_list, filename, current_step):
    fig_size = (10, 8)
    fig = plt.figure(0)
    fig.set_size_inches(fig_size[0], fig_size[1])

    ax = Axes3D(fig)

    set_ax_parameter(ax)

    plt.grid(alpha=0.2)

    draw_traj_3d(ax, agents_info, obstacles_info, agents_traj_list, step_num_list, current_step)
    plt.savefig(filename, bbox_inches="tight")

    if current_step == 0: plt.show()
    if current_step == max(step_num_list): plt.show()
    plt.close()


def plot_episode(obstacles_info, agents_info, traj_list, step_num_list, plot_save_dir, base_fig_name, last_fig_name):
    current_step = 0
    num_agents = len(step_num_list)
    total_step = max(step_num_list)
    print('num_agents:', num_agents, 'total_step:', total_step)
    while current_step < total_step:
        fig_name = base_fig_name + "_{:05}".format(current_step) + '.png'
        filename = plot_save_dir + fig_name
        plot_save_one_pic(obstacles_info, agents_info, traj_list, step_num_list, filename, current_step)
        print(filename)
        current_step += 3
    filename = plot_save_dir + last_fig_name
    plot_save_one_pic(obstacles_info, agents_info, traj_list, step_num_list, filename, total_step)


def get_cmap(N):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color
