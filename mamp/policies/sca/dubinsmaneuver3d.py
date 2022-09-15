"""
@ Author: Gang Xu
@ Date: 2021.8.13
@ Details: 3D dubins path planner
@ reference: 2020 ICRA Minimal 3D Dubins Path with Bounded Curvature and Pitch Angle
@ github: https://github.com/comrob/Dubins3D.jl
"""

import numpy as np
import math
import matplotlib.pyplot as plt

from mamp.policies.sca.dubinsmaneuver2d import dubins_path_planning, get_coordinates, plot_arrow


class DubinsManeuver3D(object):
    def __init__(self, qi, qf, Rmin, pitchlims):
        self.qi = qi
        self.qf = qf
        self.Rmin = Rmin
        self.pitchlims = pitchlims
        self.px = []
        self.py = []
        self.pz = []
        self.psi = []
        self.gamma = []
        self.maneuvers2d = []
        self.length = -1.0
        self.mode = None
        self.path = []
        self.sampling_size = 0.1


def dubinsmaneuver3d(qi, qf, Rmin, pitchlims):
    """
        Dubins3D path plannner

        input:qi, qf, Rmin, pitchlims
            qi[0]           x position of start point [m]
            qi[1]           y position of start point [m]
            qi[2]           z position of start point [m]
            qi[3]           yaw heading angle of start point [rad]
            qi[4]           pitch pitch angle of start point [rad]
            qf[0]           x position of end point [m]
            qf[1]           y position of end point [m]
            qf[2]           z position of end point [m]
            qf[3]           yaw heading angle of end point [rad]
            qf[4]           pitch pitch angle of end point [rad]
            Rmin            radius [m]
            pitchlims[0]    min pitch angle [rad]
            pitchlims[1]    max pitch angle [rad]

        output: maneuver
            maneuver.px             x position sets [m]
            maneuver.py             y position sets [m]
            maneuver.pz             z position sets [m]
            maneuver.psi            heading angle sets [rad]
            maneuver.gamma          pitch angle sets [rad]
            maneuver.maneuvers2d    horizontal and vertical dubins
            maneuver.length         length of 3D dubins
            maneuver.mode           mode of 3D dubins
            maneuver.path           coordinates of 3D dubins
        """
    maneuver3d = DubinsManeuver3D(qi, qf, Rmin, pitchlims)
    # Delta Z (height)
    zi = qi[2]
    zf = qf[2]
    dz = zf - zi

    # Multiplication factor of Rmin in [1, 1000]
    a = 1.0
    b = 1.0

    fa = try_to_construct(maneuver3d, maneuver3d.Rmin * a)
    fb = try_to_construct(maneuver3d, maneuver3d.Rmin * b)
    while len(fb) < 2:
        b *= 2.0
        fb = try_to_construct(maneuver3d, maneuver3d.Rmin * b)

    if len(fa) > 0:
        maneuver3d.maneuvers2d = fa
    else:
        if len(fb) < 2:
            print("No maneuver exists")

    # Local optimization between horizontal and vertical radii
    step = 0.1
    while abs(step) > 1e-10:
        c = b + step
        if c < 1.0:
            c = 1.0
        fc = try_to_construct(maneuver3d, maneuver3d.Rmin * c)
        if len(fc) > 0:
            if fc[1].length < fb[1].length:  # length
                b = c
                fb = fc
                step *= 2.
                continue
        step *= -0.1

    maneuver3d.maneuvers2d = fb
    maneuver3d.length = fb[1].length
    mode3d = ""
    mode = fb[0].mode + fb[1].mode
    mode3d = mode3d.join(mode)
    maneuver3d.mode = mode3d
    # draw_dubins2d(fb[0])
    # draw_dubins2d(fb[1])
    # print(mode3d)

    compute_sampling(maneuver3d, sampling_size=0.1)    # get coordinate

    return maneuver3d


def compute_sampling(maneuver3d, sampling_size=np.deg2rad(6.0)):
    maneuver_h, maneuver_v = maneuver3d.maneuvers2d
    if maneuver3d.length > 100:
        sampling_size = maneuver3d.length / 1000
    # Sample points on the final path
    maneuver3d.sampling_size = sampling_size
    qi = maneuver3d.qi
    for ran in np.arange(0, maneuver3d.length + sampling_size, sampling_size):
        offset = ran
        qSZ = get_coordinates(maneuver_v, offset)
        qXY = get_coordinates(maneuver_h, qSZ[0])
        maneuver3d.px.append(qXY[0] + qi[0])
        maneuver3d.py.append(qXY[1] + qi[1])
        maneuver3d.pz.append(qSZ[1] + qi[2])
        maneuver3d.psi.append(qXY[2])
        maneuver3d.gamma.append(qSZ[2])
        maneuver3d.path.append([qXY[0] + qi[0], qXY[1] + qi[1], qSZ[1] + qi[2], qXY[2], qSZ[2]])


def try_to_construct(maneuver3d, horizontal_radius):
    qi2D = [maneuver3d.qi[0], maneuver3d.qi[1], maneuver3d.qi[3]]  # x, y, yaw
    qf2D = [maneuver3d.qf[0], maneuver3d.qf[1], maneuver3d.qf[3]]  # x, y, yaw

    maneuver_h = dubins_path_planning(qi2D, qf2D, horizontal_radius)

    # After finding a long enough 2D curve, calculate the Dubins path on SZ axis
    qi3D = [0.0, maneuver3d.qi[2], maneuver3d.qi[4]]
    qf3D = [maneuver_h.length, maneuver3d.qf[2], maneuver3d.qf[4]]

    Rmin = maneuver3d.Rmin
    vertical_curvature = math.sqrt(1.0 / Rmin ** 2 - 1.0 / horizontal_radius ** 2)
    if vertical_curvature < 1e-5:
        return []
    vertical_radius = 1.0 / vertical_curvature
    maneuver_v = dubins_path_planning(qi3D, qf3D, vertical_radius)

    case = ""
    case = case.join(maneuver_v.mode)
    if case == "RLR" or case == "RLR":
        return []
    if case[0] == "R":
        if maneuver3d.qi[4] - maneuver_v.t < maneuver3d.pitchlims[0]:
            return []
    else:
        if maneuver3d.qi[4] + maneuver_v.t > maneuver3d.pitchlims[1]:
            return []
    return [maneuver_h, maneuver_v]  # Final 3D path is formed by the two curves (maneuver_h, maneuver_v)


def set_ax(ax):
    ax.set_aspect('auto')
    ax.set_xlabel('X(m)')
    ax.set_xlim(-22, 25)
    ax.set_ylabel('Y(m)')
    ax.set_ylim(-22, 25)
    ax.set_zlabel('Z(m)')
    ax.set_zlim(-5, 50)


def draw_dubins3d(px_sets, py_sets, pz_sets):
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot3D(px_sets, py_sets, pz_sets, 'red')
    set_ax(ax)
    plt.show()


def draw_dubins2d(maneuver2d):
    plt.plot(maneuver2d.px, maneuver2d.py, label="final course " + "".join(maneuver2d.mode))
    # plotting
    plot_arrow(maneuver2d.qi[0], maneuver2d.qi[1], maneuver2d.qi[2])
    plot_arrow(maneuver2d.qf[0], maneuver2d.qf[1], maneuver2d.qf[2])
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    print("3D-Dubins Planner Start!!")
    """
    3D-Dubins path plannner

    input:
        start_x : x position of start point [m]
        start_y : y position of start point [m]
        start_z : z position of start point [m]
        start_psi : heading angle of start point [rad]
        start_gamma : flight path angle of start point [rad]

        end_x : x position of end point [m]
        end_y : y position of end point [m]
        end_z : z position of end point [m]
        end_psi : heading angle of end point [rad]
        end_gamma : flight path angle of end point [rad]

        Rmin : minimum turning radius [m]

    output:
        px : x coordinates of path
        py : y coordinates of path
        pz : z coordinates of path
        ppsi : heading angle of path points
        pgamma: flight path angle of path points
        mode : type of curve

    """
    n = 4
    k = 45
    test_qi = [[0.0, 0.0, 3.0, np.deg2rad(-90), np.deg2rad(0.0)], [0.0, 0.0, 13.0, np.deg2rad(-90), np.deg2rad(0.0)]]
    test_qf = [[0.0, 0.0, 13.0, np.deg2rad(90), np.deg2rad(0.0)], [0.0, 0.0, 3.0, np.deg2rad(90), np.deg2rad(0.0)]]
    test_pitchlims = [np.deg2rad(-45.0), np.deg2rad(45.0)]
    test_Rmin = 1.5

    '''instance from paper is true where the length is 976.79'''
    # test_qi = [-80.0, 10.0, 250.0, np.deg2rad(20.0), np.deg2rad(0.0)]
    # test_qf = [50.0, 70.0, 0.0, np.deg2rad(240.0), np.deg2rad(0.0)]
    # test_pitchlims = [np.deg2rad(-15.0), np.deg2rad(20.0)]
    # test_Rmin = 40

    try:
        maneuver = dubinsmaneuver3d(test_qi[0], test_qf[0], test_Rmin, test_pitchlims)
        maneuver1 = dubinsmaneuver3d(test_qi[1], test_qf[1], test_Rmin, test_pitchlims)
        print("Trajectory Type ---->", maneuver.mode, " and Length ----->", maneuver.length)

        print(len(maneuver.path), maneuver.path)
        print(len(maneuver1.path), maneuver1.path)
        # print(maneuver.path[75])
        draw_dubins3d(maneuver.px, maneuver.py, maneuver.pz)
        draw_dubins3d(maneuver1.px, maneuver1.py, maneuver1.pz)

    except:
        print("NOT POSSIBLE")
