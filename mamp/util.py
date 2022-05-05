import numpy as np
from math import sqrt, asin, acos, atan2, pi, floor
eps = 10 ** 5       # Keep 5 decimal.


def satisfied_constraint(agent, vCand):
    vA = agent.vel_global_frame
    next_pA = agent.pos_global_frame + agent.timeStep * vCand  # Constraints of z-plane.
    # if next_pA[2] < 0.0:
    #     print('---------------------------------- z-Axis < 0.0 is ture', next_pA[2])
    costheta = np.dot(vA, vCand) / (np.linalg.norm(vA) * np.linalg.norm(vCand))
    if costheta > 1.0:
        costheta = 1.0
    elif costheta < -1.0:
        costheta = -1.0
    theta = acos(costheta)  # Rotational constraints.
    if theta <= agent.max_heading_change and next_pA[2] >= 0.0:
        return True
    else:
        return False


def reached(p1, p2, bound=0.5):
    if l3norm(p1, p2) < bound:
        return True
    else:
        return False


def is_intersect(pA, pB, combined_radius, v_dif):
    pAB = pB - pA
    dist_pAB = np.linalg.norm(pAB)
    if dist_pAB <= combined_radius:
        dist_pAB = combined_radius
    theta_pABBound = asin(combined_radius / dist_pAB)
    theta_pABvCand = acos(np.dot(pAB, v_dif) / (dist_pAB * np.linalg.norm(v_dif)))
    if theta_pABBound <= theta_pABvCand:  # No intersecting or tangent.

        return False
    else:
        return True


def cartesian2spherical(agent, vA_post):
    speed = l3norm(vA_post, [0, 0, 0])
    if speed < 0.001:
        alpha = 0.0
        beta = 0.0
        gamma = 0.0
    else:
        alpha = atan2(vA_post[1], vA_post[0]) - agent.heading_global_frame[0]
        beta = atan2(vA_post[2], sqrt(pow(vA_post[0], 2) + pow(vA_post[1], 2))) - agent.heading_global_frame[1]
        gamma = 0.0
    action = [vA_post[0], vA_post[1], vA_post[2], speed, alpha, beta, gamma]
    return action


def det3order(a, b, c):
    a1 = a[0]*b[1]*c[2]
    a2 = a[1]*b[2]*c[0]
    a3 = a[2]*b[0]*c[1]
    b1 = a[2]*b[1]*c[0]
    b2 = a[1]*b[0]*c[2]
    b3 = a[0]*b[2]*c[1]
    return a1 + a2 + a3 - b1 - b2 - b3


def length_vec(vec):
    return np.linalg.norm(vec)


def normalize(vec):
    return np.array(vec) / np.linalg.norm(vec)


def cross(v1, v2):
    return np.array([v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]])


def takeSecond(elem):
    return elem[1]


def sqr(a):
    return a ** 2


def absSq(vec):
    return np.dot(vec, vec)


def l2norm(x, y):
    return round(sqrt(l2normsq(x, y)), 5)


def l2normsq(x, y):
    return round((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2, 5)


def l3normsq(x, y):
    return round((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2, 5)


def l3norm(p1, p2):
    """ Compute Euclidean distance for 3D """
    return round(sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2), 5)


def pi_2_pi(angle):  # to -pi - pi
    return (angle + pi) % (2 * pi) - pi


def mod2pi(theta):  # to 0 - 2*pi
    return theta - 2.0 * pi * floor(theta / 2.0 / pi)


def leftOf(a, b, c):
    return det(a - c, b - a)


def det(p, q):
    return p[0] * q[1] - p[1] * q[0]


def is_parallel(vec1, vec2):
    """ Whether two three-dimensional vectors are parallel """
    assert vec1.shape == vec2.shape, r'Input parameter shape must be the same'
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    vec1_normalized = vec1 / norm_vec1
    vec2_normalized = vec2 / norm_vec2
    if norm_vec1 <= 1e-5 or norm_vec2 <= 1e-5:
        return True
    elif round(1.0 - abs(np.dot(vec1_normalized, vec2_normalized)), 5) < 3e-3:
        return True
    else:
        return False


def distance(p1, p2):
    """ Compute Euclidean distance for 3D """
    return round(sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) + 1e-5, 5)


def get_phi(vec):  # Compute the angle between two vectors after projection on the XY plane.
    if vec[1] >= 0:
        phi = atan2(vec[1], vec[0])
    else:
        phi = 2 * pi + atan2(vec[1], vec[0])
    return int(phi * eps) / eps



