"""
@ Author: Gang Xu
@ Date: 2021.8.13
@ Details: 2D dubins path planner
@ reference: 2020 ICRA Minimal 3D Dubins Path with Bounded Curvature and Pitch Angle
@ github: https://github.com/comrob/Dubins3D.jl
"""


import math
import numpy as np
from mamp.util import mod2pi, pi_2_pi
import matplotlib.pyplot as plt


class DubinsManeuver(object):
    def __init__(self, qi, qf, r_min):
        self.qi = qi
        self.qf = qf
        self.r_min = r_min
        self.t = -1.0
        self.p = -1.0
        self.q = -1.0
        self.px = []
        self.py = []
        self.pyaw = []
        self.path = []
        self.mode = None
        self.length = -1.0
        self.sampling_size = None


def LSL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d + sa - sb

    mode = ["L", "S", "L"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((cb - ca), tmp0)
    t = mod2pi(-alpha + tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(beta - tmp1)

    return t, p, q, mode


def RSR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d - sa + sb
    mode = ["R", "S", "R"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((ca - cb), tmp0)
    t = mod2pi(alpha - tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(-beta + tmp1)

    return t, p, q, mode


def LSR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(-mod2pi(beta) + tmp2)

    return t, p, q, mode


def RSL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(beta - tmp2)

    return t, p, q, mode


def RLR(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["R", "L", "R"]
    tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp_rlr) > 1.0:
        return None, None, None, mode

    p = mod2pi(2 * math.pi - math.acos(tmp_rlr))
    t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
    q = mod2pi(alpha - beta - t + mod2pi(p))
    return t, p, q, mode


def LRL(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["L", "R", "L"]
    tmp_lrl = (6. - d * d + 2 * c_ab + 2 * d * (- sa + sb)) / 8.
    if abs(tmp_lrl) > 1:
        return None, None, None, mode
    p = mod2pi(2 * math.pi - math.acos(tmp_lrl))
    t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.)
    q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))

    return t, p, q, mode


def dubins_path_planning_from_origin(ex, ey, syaw, eyaw, c):
    # nomalize
    dx = ex
    dy = ey
    D = math.sqrt(dx ** 2.0 + dy ** 2.0)
    d = D / c

    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(syaw - theta)
    beta = mod2pi(eyaw - theta)

    planners = [LSL, RSR, LSR, RSL, RLR, LRL]

    bcost = float("inf")
    bt, bp, bq, bmode = None, None, None, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)
        if t is None:
            continue

        cost = c*(abs(t) + abs(p) + abs(q))
        if bcost > cost:
            bt, bp, bq, bmode = t, p, q, mode
            bcost = cost

    px, py, pyaw = generate_course([bt, bp, bq], bmode, c)

    return px, py, pyaw, bmode, bcost, bt, bp, bq


def dubins_path_planning(start_pos, end_pos, c):
    """
    Dubins path plannner

    input:start_pos, end_pos, c
        start_pos[0]    x position of start point [m]
        start_pos[1]    y position of start point [m]
        start_pos[2]    yaw angle of start point [rad]
        end_pos[0]      x position of end point [m]
        end_pos[1]      y position of end point [m]
        end_pos[2]      yaw angle of end point [rad]
        c               radius [m]

    output: maneuver
        maneuver.t              the first segment curve of dubins
        maneuver.p              the second segment line of dubins
        maneuver.q              the third segment curve of dubins
        maneuver.px             x position sets [m]
        maneuver.py             y position sets [m]
        maneuver.pyaw           heading angle sets [rad]
        maneuver.length         length of dubins
        maneuver.mode           mode of dubins
    """
    maneuver = DubinsManeuver(start_pos, end_pos, c)
    sx, sy = start_pos[0], start_pos[1]
    ex, ey = end_pos[0], end_pos[1]
    syaw, eyaw = start_pos[2], end_pos[2]

    ex = ex - sx
    ey = ey - sy

    lpx, lpy, lpyaw, mode, clen, t, p, q = dubins_path_planning_from_origin(ex, ey, syaw, eyaw, c)

    px = [math.cos(-syaw) * x + math.sin(-syaw) * y + sx for x, y in zip(lpx, lpy)]
    py = [-math.sin(-syaw) * x + math.cos(-syaw) * y + sy for x, y in zip(lpx, lpy)]
    pyaw = [pi_2_pi(iyaw + syaw) for iyaw in lpyaw]
    maneuver.t, maneuver.p, maneuver.q, maneuver.mode  = t, p, q, mode
    maneuver.px, maneuver.py, maneuver.pyaw,  maneuver.length = px, py, pyaw, clen

    return maneuver


def generate_course(length, mode, c):
    px = [0.0]
    py = [0.0]
    pyaw = [0.0]

    for m, l in zip(mode, length):
        pd = 0.0
        if m == "S":
            d = 1.0 / c
        else:  # turning course
            d = np.deg2rad(6.0)

        while pd < abs(l - d):
            px.append(px[-1] + d * c * math.cos(pyaw[-1]))
            py.append(py[-1] + d * c * math.sin(pyaw[-1]))

            if m == "L":  # left turn
                pyaw.append(pyaw[-1] + d)
            elif m == "S":  # Straight
                pyaw.append(pyaw[-1])
            elif m == "R":  # right turn
                pyaw.append(pyaw[-1] - d)
            pd += d

        d = l - pd
        px.append(px[-1] + d * c * math.cos(pyaw[-1]))
        py.append(py[-1] + d * c * math.sin(pyaw[-1]))

        if m == "L":  # left turn
            pyaw.append(pyaw[-1] + d)
        elif m == "S":  # Straight
            pyaw.append(pyaw[-1])
        elif m == "R":  # right turn
            pyaw.append(pyaw[-1] - d)
        pd += d

    return px, py, pyaw


def get_coordinates(maneuver, offset):
    noffset = offset / maneuver.r_min
    qi = [0., 0., maneuver.qi[2]]

    l1 = maneuver.t
    l2 = maneuver.p
    q1 = get_position_in_segment(l1, qi, maneuver.mode[0])      # Final do segmento 1
    q2 = get_position_in_segment(l2, q1, maneuver.mode[1])       # Final do segmento 2

    if noffset < l1:
        q = get_position_in_segment(noffset, qi, maneuver.mode[0])
    elif noffset < (l1 + l2):
        q = get_position_in_segment(noffset - l1, q1, maneuver.mode[1])
    else:
        q = get_position_in_segment(noffset - l1 - l2, q2, maneuver.mode[2])

    q[0] = q[0] * maneuver.r_min + qi[0]
    q[1] = q[1] * maneuver.r_min + qi[1]
    q[2] = mod2pi(q[2])

    return q


def get_position_in_segment(offset, qi, mode):
    q = [0.0, 0.0, 0.0]
    if mode == 'L':
        q[0] = qi[0] + math.sin(qi[2] + offset) - math.sin(qi[2])
        q[1] = qi[1] - math.cos(qi[2] + offset) + math.cos(qi[2])
        q[2] = qi[2] + offset
    elif mode == 'R':
        q[0] = qi[0] - math.sin(qi[2] - offset) + math.sin(qi[2])
        q[1] = qi[1] + math.cos(qi[2] - offset) - math.cos(qi[2])
        q[2] = qi[2] - offset
    elif mode == 'S':
        q[0] = qi[0] + math.cos(qi[2]) * offset
        q[1] = qi[1] + math.sin(qi[2]) * offset
        q[2] = qi[2]
    return q


def get_sampling_points(maneuver, sampling_size=0.1):
    points = []
    for offset in np.arange(0.0, maneuver.length+sampling_size, sampling_size):
        points.append(get_coordinates(maneuver, offset))
    return points


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


if __name__ == '__main__':
    print("Dubins Path Planner Start!!")

    test_qi = [[0.0, 5.0, np.deg2rad(180.0)], [10.0, 5.0, np.deg2rad(180.0)]]        # start_x, start_y, start_yaw
    test_qf = [[10.0, 5.0, np.deg2rad(180.0)], [0.0, 5.0, np.deg2rad(0.0)]]       # end_x, end_y, end_yaw

    rmin = 1.5

    test_maneuver = dubins_path_planning(test_qi[0], test_qf[0], rmin)
    test_maneuver1 = dubins_path_planning(test_qi[1], test_qf[1], rmin)

    path = []
    for i in range(len(test_maneuver.px)):
        path.append([test_maneuver.px[i], test_maneuver.py[i], test_maneuver.pyaw[i]])
    print(len(path), test_maneuver.length, path)

    plt.plot(test_maneuver.px, test_maneuver.py, label="test_maneuver " + "".join(test_maneuver.mode))
    # plt.plot(test_maneuver1.px, test_maneuver1.py, label="test_maneuver1 " + "".join(test_maneuver1.mode))


    # plotting
    plot_arrow(test_qi[0][0], test_qi[0][1], test_qi[0][2])
    plot_arrow(test_qf[0][0], test_qf[0][1], test_qf[0][2])
    # plot_arrow(test_qi[1][0], test_qi[1][1], test_qi[1][2])
    # plot_arrow(test_qf[1][0], test_qf[1][1], test_qf[1][2])

    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
