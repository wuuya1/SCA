import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from math import sin, cos, atan2, sqrt, pow


# models
def get_car_model(length=2.0, width=1.5, height=1.0):
    car_model = [[-width / 2., length / 2, height / 2], [-width / 2, -length / 2, height / 2],
                 [width / 2., -length / 2, height / 2], [width / 2, length / 2, height / 2],
                 [-width / 2., length / 2, -height / 2], [-width / 2, -length / 2, -height / 2],
                 [width / 2., -length / 2, -height / 2], [width / 2, length / 2, -height / 2],
                 ]
    return car_model


def get_uav_model(uav_radius, tall=0.05):
    length = 1.6 * uav_radius
    width = 1.1 * uav_radius
    uav_model = [[0., length - 0.5 * width, tall], [-0.5 * width, -0.5 * width, tall], [0., -0.2 * width, tall],
                 [0.5 * width, -0.5 * width, tall],
                 [0., length - 0.5 * width, -tall], [-0.5 * width, -0.5 * width, -tall], [0., -0.2 * width, -tall],
                 [0.5 * width, -0.5 * width, -tall]]
    return uav_model


def get_uav_goal_posture(length=0.9, width=0.6, tall=0.05):
    uav_model = [[0., length - 0.5 * width, tall], [-0.5 * width, -0.5 * width, tall], [0., -0.2 * width, tall],
                 [0.5 * width, -0.5 * width, tall],
                 [0., length - 0.5 * width, -tall], [-0.5 * width, -0.5 * width, -tall], [0., -0.2 * width, -tall],
                 [0.5 * width, -0.5 * width, -tall]]
    return uav_model


def get_cube_model(length=2.0, width=1.5, height=1.0):
    cube_model = [[-length / 2., width / 2, height / 2], [-length / 2, -width / 2, height / 2],
                  [length / 2., -width / 2, height / 2], [length / 2, width / 2, height / 2],
                  [-length / 2., width / 2, -height / 2], [-length / 2, -width / 2, -height / 2],
                  [length / 2., -width / 2, -height / 2], [length / 2, width / 2, -height / 2],
                  ]
    return cube_model


def get_building_model(long=10.0, width=10.0, tall=10.0):
    building_model = [[-long / 2, width / 2, tall], [-long / 2, -width / 2, tall], [long / 2, -width / 2, tall],
                      [long / 2, width / 2, tall],
                      [-long / 2, width / 2, 0.0], [-long / 2, -width / 2, 0.0], [long / 2, -width / 2, 0.0],
                      [long / 2, width / 2, 0.0]]
    return building_model


def draw_env(ax, buildings_obj_list, keypoints_obj_list, connection_matrix):
    pass


def draw(origin_pos, objects_list, buildings_obj_list, keypoints_obj_list, connection_matrix):
    fig = plt.figure()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim([-500.0, 3500.0])
    plt.ylim([-1000.0, 1000.0])
    ax = fig.add_subplot(1, 1, 1)

    ax.set_aspect('equal')

    draw_objects(ax, objects_list)
    draw_buildings(ax, origin_pos, buildings_obj_list)
    draw_roads(ax, keypoints_obj_list, connection_matrix)
    plt.show()


def draw_buildings(ax, origin_pos, buildings_obj):
    for building_obj in buildings_obj:
        id = building_obj.id
        pos = building_obj.pos
        draw_rectangle(ax, origin_pos, pos)


def draw_roads(ax, keypoints_obj_list, connection_matrix):
    for index_i, info in enumerate(connection_matrix):
        self_pos = keypoints_obj_list[index_i].pos
        for index_j, distance in enumerate(info):
            if distance > 0:
                target_pos = keypoints_obj_list[index_j].pos
                x = [self_pos[0], target_pos[0]]
                y = [self_pos[1], target_pos[1]]
                plt.plot(x, y, color='r')
                plt.scatter(x, y, color='b')


def draw_objects(ax, objects_obj):
    for object_obj in objects_obj:
        pass


def draw_rectangle(ax, origin_pos, pos):
    pos_x = pos[0]
    pos_y = pos[1]
    ax.add_patch(
        plt.Rectangle(
            (pos_x - 5, pos_y - 5),
            10,  # width
            10,  # height
            color='maroon',
            alpha=0.5
        ))


def draw_agent_3d(ax, pos_global_frame, goal, c_map, heading_global_frame, my_agent_model, a_id, color='blue'):
    pos = pos_global_frame
    agent_model = my_agent_model
    convert_to_actual_model(agent_model, pos_global_frame, heading_global_frame)
    num_corner_point_per_layer = int(len(agent_model) / 2)
    x_list = []
    y_list = []
    z_list = []
    for layer in range(2):
        x_list.clear(), y_list.clear(), z_list.clear()
        for i in range(num_corner_point_per_layer):
            x_list.append(agent_model[i + layer * num_corner_point_per_layer][0])
            y_list.append(agent_model[i + layer * num_corner_point_per_layer][1])
            z_list.append(agent_model[i + layer * num_corner_point_per_layer][2])
        pannel = [list(zip(x_list, y_list, z_list))]
        ax.add_collection3d(Poly3DCollection(pannel, facecolors='goldenrod', alpha=0.9))

    for i in range(num_corner_point_per_layer - 1):
        x_list.clear(), y_list.clear(), z_list.clear()
        if i == 0:
            x_list.append(agent_model[0][0])
            x_list.append(agent_model[num_corner_point_per_layer - 1][0])
            x_list.append(agent_model[2 * num_corner_point_per_layer - 1][0])
            x_list.append(agent_model[num_corner_point_per_layer][0])
            y_list.append(agent_model[0][1])
            y_list.append(agent_model[num_corner_point_per_layer - 1][1])
            y_list.append(agent_model[2 * num_corner_point_per_layer - 1][1])
            y_list.append(agent_model[num_corner_point_per_layer][1])
            z_list.append(agent_model[0][2])
            z_list.append(agent_model[num_corner_point_per_layer - 1][2])
            z_list.append(agent_model[2 * num_corner_point_per_layer - 1][2])
            z_list.append(agent_model[num_corner_point_per_layer][2])
            pannel = [list(zip(x_list, y_list, z_list))]
            ax.add_collection3d(Poly3DCollection(pannel, facecolors=color, alpha=0.9))

        x_list.clear(), y_list.clear(), z_list.clear()
        x_list.append(agent_model[i][0])
        x_list.append(agent_model[i + 1][0])
        x_list.append(agent_model[i + 1 + num_corner_point_per_layer][0])
        x_list.append(agent_model[i + num_corner_point_per_layer][0])
        y_list.append(agent_model[i][1])
        y_list.append(agent_model[i + 1][1])
        y_list.append(agent_model[i + 1 + num_corner_point_per_layer][1])
        y_list.append(agent_model[i + num_corner_point_per_layer][1])
        z_list.append(agent_model[i][2])
        z_list.append(agent_model[i + 1][2])
        z_list.append(agent_model[i + 1 + num_corner_point_per_layer][2])
        z_list.append(agent_model[i + num_corner_point_per_layer][2])
        pannel = [list(zip(x_list, y_list, z_list))]

        ax.text(pos[0] - 0.1, pos[1] - 0.1, pos[2] + 0.3, r'$%s$' % a_id, fontsize=10, fontweight='bold', zorder=3)
        ax.plot([goal[0]], [goal[1]], [goal[2]], '*', color=c_map, markersize=10, linewidth=3.0)
        ax.add_collection3d(Poly3DCollection(pannel, facecolors=color, alpha=0.9))  # , alpha=0.7


def draw_goal_posture(ax, goal, goal_heading_frame, my_agent_model, a_id, color='red'):
    agent_model = my_agent_model
    convert_to_actual_model(agent_model, goal, goal_heading_frame)
    num_corner_point_per_layer = int(len(agent_model) / 2)
    x_list = []
    y_list = []
    z_list = []
    for layer in range(2):
        x_list.clear(), y_list.clear(), z_list.clear()
        for i in range(num_corner_point_per_layer):
            x_list.append(agent_model[i + layer * num_corner_point_per_layer][0])
            y_list.append(agent_model[i + layer * num_corner_point_per_layer][1])
            z_list.append(agent_model[i + layer * num_corner_point_per_layer][2])
        pannel = [list(zip(x_list, y_list, z_list))]
        ax.add_collection3d(Poly3DCollection(pannel, facecolors='blue', alpha=0.9))

    for i in range(num_corner_point_per_layer - 1):
        x_list.clear(), y_list.clear(), z_list.clear()
        if i == 0:
            x_list.append(agent_model[0][0])
            x_list.append(agent_model[num_corner_point_per_layer - 1][0])
            x_list.append(agent_model[2 * num_corner_point_per_layer - 1][0])
            x_list.append(agent_model[num_corner_point_per_layer][0])
            y_list.append(agent_model[0][1])
            y_list.append(agent_model[num_corner_point_per_layer - 1][1])
            y_list.append(agent_model[2 * num_corner_point_per_layer - 1][1])
            y_list.append(agent_model[num_corner_point_per_layer][1])
            z_list.append(agent_model[0][2])
            z_list.append(agent_model[num_corner_point_per_layer - 1][2])
            z_list.append(agent_model[2 * num_corner_point_per_layer - 1][2])
            z_list.append(agent_model[num_corner_point_per_layer][2])
            pannel = [list(zip(x_list, y_list, z_list))]
            ax.add_collection3d(Poly3DCollection(pannel, facecolors=color, alpha=0.9))

        x_list.clear(), y_list.clear(), z_list.clear()
        x_list.append(agent_model[i][0])
        x_list.append(agent_model[i + 1][0])
        x_list.append(agent_model[i + 1 + num_corner_point_per_layer][0])
        x_list.append(agent_model[i + num_corner_point_per_layer][0])
        y_list.append(agent_model[i][1])
        y_list.append(agent_model[i + 1][1])
        y_list.append(agent_model[i + 1 + num_corner_point_per_layer][1])
        y_list.append(agent_model[i + num_corner_point_per_layer][1])
        z_list.append(agent_model[i][2])
        z_list.append(agent_model[i + 1][2])
        z_list.append(agent_model[i + 1 + num_corner_point_per_layer][2])
        z_list.append(agent_model[i + num_corner_point_per_layer][2])
        pannel = [list(zip(x_list, y_list, z_list))]
        ax.text(goal[0] - 0.1, goal[1] - 0.1, goal[2] + 0.3, r'$%s$' % a_id, fontsize=10, fontweight='bold', zorder=3)
        ax.add_collection3d(Poly3DCollection(pannel, facecolors=color, alpha=0.9))


def draw_sphere(ax, obstacle_dict):
    center = obstacle_dict['position']
    radius = obstacle_dict['feature']
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, rstride=4, cstride=8, color='LightGray')


def CreateSphere(center, r):
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    x, y, z = r * x + center[0], r * y + center[1], r * z + center[2]
    return x, y, z


def draw_Spheres(ax, pos, radius, c_map):
    (xs, ys, zs) = CreateSphere(pos, radius)
    ax.plot_wireframe(xs, ys, zs, alpha=0.15, color=c_map)


def draw_agent(ax, pos, goal, vel, radius, c_map, a_id):
    draw_Spheres(ax, pos, radius, c_map)
    ax.quiver(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], length=1, color=c_map)
    ax.text(pos[0] - 0.1, pos[1] - 0.1, pos[2] + 0.3, r'$%s$' % a_id, fontsize=10, fontweight='bold', zorder=3)
    ax.plot([goal[0]], [goal[1]], [goal[2]], '*', color=c_map, markersize=10, linewidth=3.0)


def convert_to_actual_model(agent_model, pos_global_frame, heading_global_frame):
    alpha = heading_global_frame[0]
    beta = heading_global_frame[1]
    gamma = heading_global_frame[2]
    for point in agent_model:
        x = point[0]
        y = point[1]
        z = point[2]
        # Compute pitch pose
        r = sqrt(pow(y, 2) + pow(z, 2))
        beta_model = atan2(z, y)
        beta_ = beta + beta_model
        y = r * cos(beta_)
        z = r * sin(beta_)
        # Compute roll pose
        h = sqrt(pow(x, 2) + pow(z, 2))
        gama_model = atan2(z, x)
        gamma_ = gamma + gama_model
        x = h * cos(gamma_)
        z = h * sin(gamma_)
        # Compute yaw pose
        l = sqrt(pow(x, 2) + pow(y, 2))
        alpha_model = atan2(y, x)
        alpha_ = alpha + alpha_model - np.pi / 2
        point[0] = l * cos(alpha_) + pos_global_frame[0]
        point[1] = l * sin(alpha_) + pos_global_frame[1]
        point[2] = z + pos_global_frame[2]
