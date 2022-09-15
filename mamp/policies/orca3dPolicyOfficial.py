"""
@ Author: Gang Xu
@ Date: 2022.04.16
@ Details: ORCA(RVO2) 3D
@ reference: Reciprocal n-Body Collision Avoidance*
@ github: https://github.com/snape/RVO2-3D
"""
import time
import numpy as np
from math import sqrt, atan2
from mamp.configs.config import eps, DT, rvo3d_epsilon
from mamp.util import distance, absSq, sqr, normalize, reached


class Line(object):
    def __init__(self):
        self.direction = np.array([0.0, 0.0, 1.0])  # The direction of the directed line.
        self.point = np.array([0.0, 0.0, 0.0])  # A point on the directed line.


class Plane(object):
    def __init__(self):
        self.point = np.array([0.0, 0.0, 0.0])  # A point on the plane.
        self.normal = np.array([0.0, 0.0, 1.0])  # The normal to the plane.


class ORCA3DPolicy(object):
    """ ORCA3DPolicy """

    def __init__(self):
        self.now_goal = None
        self.update_nowGoal_dist = 1.0
        self.orcaPlanes = []
        self.type = "internal"
        self.new_velocity = None

    def find_next_action(self, dict_comm, agent, kdTree):
        """
        Function: ORCA3D compute suitable speed for agents
        """
        start_t = time.time()

        self.orcaPlanes.clear()
        invTimeHorizon = 1.0 / agent.timeHorizon
        self.new_velocity = np.array([0.0, 0.0, 0.0])

        self.get_trajectory(agent)
        v_pref = compute_v_pref(self.now_goal, agent)

        computeNeighbors(agent, kdTree)

        if distance(agent.vel_global_frame, [0.0, 0.0, 0.0]) <= 1e-5:  # the first step is required for dynamic
            vA_post = 0.3 * v_pref

        else:
            for obj in agent.neighbors:
                obj = obj[0]
                relativePosition = obj.pos_global_frame - agent.pos_global_frame
                relativeVelocity = agent.vel_global_frame - obj.vel_global_frame
                distSq = absSq(relativePosition)
                agent_rad = agent.radius + 0.05
                obj_rad = obj.radius + 0.05
                combinedRadius = agent_rad + obj_rad
                combinedRadiusSq = sqr(combinedRadius)

                plane = Plane()
                if distSq > combinedRadiusSq:  # No collision.
                    # Vector from cutoff center to relative velocity.
                    w = relativeVelocity - invTimeHorizon * relativePosition
                    wLengthSq = absSq(w)
                    dotProduct = np.dot(w, relativePosition)
                    if dotProduct < 0.0 and sqr(dotProduct) > combinedRadiusSq * wLengthSq:
                        # Project on cut-off circle.
                        wLength = sqrt(wLengthSq)
                        unitW = w / wLength

                        plane.normal = unitW
                        u = (combinedRadius * invTimeHorizon - wLength) * unitW
                        # print("Project on cut-off circle.")
                    else:
                        # Project on cone.
                        difSq = distSq - combinedRadiusSq
                        dot_product = np.dot(relativePosition, relativeVelocity)
                        wwSq = absSq(np.cross(relativePosition, relativeVelocity)) / difSq  # 半径的平方
                        pApBLength = np.linalg.norm(relativePosition)
                        pAp1Length = dot_product / pApBLength  # p1是vA-vB到直线pB-pA的垂足点
                        p1otLength = sqrt(wwSq) * (combinedRadius / pApBLength)  # ot是对应的圆心
                        pAotlength = pAp1Length + p1otLength

                        t = pAotlength / pApBLength
                        ww = relativeVelocity - t * relativePosition
                        wwLength = np.linalg.norm(ww)
                        unitWW = ww / wwLength
                        plane.normal = unitWW
                        u = (combinedRadius * t - wwLength) * unitWW
                        # print("Project on cone. t = ", t, R1, RR)
                else:  # Collision.
                    invTimeStep = 1.0 / agent.timeStep
                    w = relativeVelocity - invTimeStep * relativePosition
                    wLength = np.linalg.norm(w)
                    unitW = w / wLength
                    plane.normal = unitW
                    u = (combinedRadius * invTimeStep - wLength) * unitW
                    # print('Collision.')
                plane.point = agent.vel_global_frame + 0.5 * u
                self.orcaPlanes.append(plane)

            planeFail = self.linearProgram3(self.orcaPlanes, agent.maxSpeed, v_pref)

            if planeFail < len(self.orcaPlanes):
                self.linearProgram4(self.orcaPlanes, planeFail, agent.maxSpeed)

            vA_post = np.array([self.new_velocity[0], self.new_velocity[1], self.new_velocity[2]])

        action = cartesian2spherical(agent, vA_post)

        end_t = time.time()
        cost_step = end_t - start_t
        agent.total_time += cost_step

        dist = round(distance(agent.pos_global_frame, agent.goal_global_frame), 5)
        print('agent' + str(agent.id), len(agent.neighbors), round(action[3], 5), '终点距离：', dist)

        return action

    def linearProgram1(self, planes, planeNo, line, maxSpeed, vel_pref, dir_opt):
        dotProduct = np.dot(line.point, line.direction)
        discriminant = sqr(dotProduct) + sqr(maxSpeed) - absSq(line.point)

        if discriminant < 0.0:
            # Max speed sphere fully invalidates line.
            return False

        sqrtDiscriminant = sqrt(discriminant)
        tLeft = -dotProduct - sqrtDiscriminant
        tRight = -dotProduct + sqrtDiscriminant

        for i in range(planeNo):
            numerator = np.dot((planes[i].point - line.point), planes[i].normal)
            denominator = np.dot(line.direction, planes[i].normal)

            if sqr(denominator) <= rvo3d_epsilon:
                # Lines line is (almost) parallel to plane i.
                if numerator > 0.0:
                    return False
                else:
                    continue

            t = numerator / denominator

            if denominator >= 0.0:
                # Plane i bounds line on the left.
                tLeft = max(tLeft, t)
            else:
                # Plane i bounds line on the right.
                tRight = min(tRight, t)

            if tLeft > tRight:
                return False

        if dir_opt:
            # Optimize direction.
            if np.dot(vel_pref, line.direction) > 0.0:
                # Take right extreme.
                self.new_velocity = line.point + tRight * line.direction

            else:
                # Take left extreme.
                self.new_velocity = line.point + tLeft * line.direction
        else:
            # Optimize closest point.
            t = np.dot(line.direction, (vel_pref - line.point))

            if t < tLeft:
                self.new_velocity = line.point + tLeft * line.direction
            elif t > tRight:
                self.new_velocity = line.point + tRight * line.direction
            else:
                self.new_velocity = line.point + t * line.direction

        return True

    def linearProgram2(self, planes, planeNo, maxSpeed, vel_pref, dir_opt):
        planeDist = np.dot(planes[planeNo].point, planes[planeNo].normal)
        planeDistSq = sqr(planeDist)
        radiusSq = sqr(maxSpeed)

        if planeDistSq > radiusSq:
            # Max speed sphere fully invalidates plane planeNo.
            return False

        planeRadiusSq = radiusSq - planeDistSq

        planeCenter = planeDist * planes[planeNo].normal

        if dir_opt:
            # Project direction opt_vel on plane planeNo.
            planeOptVelocity = vel_pref - np.dot(vel_pref, planes[planeNo].normal) * planes[planeNo].normal
            planeOptVelocityLengthSq = absSq(planeOptVelocity)

            if planeOptVelocityLengthSq <= rvo3d_epsilon:
                self.new_velocity = planeCenter
            else:
                self.new_velocity = planeCenter + sqrt(planeRadiusSq / planeOptVelocityLengthSq) * planeOptVelocity

        else:
            # Project point optVelocity on plane planeNo.
            dot_product = np.dot((planes[planeNo].point - vel_pref), planes[planeNo].normal)
            self.new_velocity = vel_pref + dot_product * planes[planeNo].normal

            # If outside planeCircle, project on planeCircle.
            if absSq(self.new_velocity) > radiusSq:
                planeResult = self.new_velocity - planeCenter
                planeResultLengthSq = absSq(planeResult)
                self.new_velocity = planeCenter + sqrt(planeRadiusSq / planeResultLengthSq) * planeResult

        for i in range(planeNo):
            if np.dot(planes[i].normal, (planes[i].point - self.new_velocity)) > 0.0:
                # Result does not satisfy constraint i.Compute new optimal result.
                # Compute intersection line of plane i and plane planeNo.
                crossProduct = np.cross(planes[i].normal, planes[planeNo].normal)

                if absSq(crossProduct) <= rvo3d_epsilon:
                    # Planes planeNo and i are (almost) parallel, and plane i fully invalidates plane planeNo.
                    return False

                line = Line()
                line.direction = normalize(crossProduct)
                lineNormal = np.cross(line.direction, planes[planeNo].normal)
                dot_product1 = np.dot((planes[i].point - planes[planeNo].point), planes[i].normal)
                dot_product2 = np.dot(lineNormal, planes[i].normal)
                line.point = planes[planeNo].point + (dot_product1 / dot_product2) * lineNormal

                if not self.linearProgram1(planes, i, line, maxSpeed, vel_pref, dir_opt):
                    return False

        return True

    def linearProgram3(self, planes, maxSpeed, vel_pref, dir_opt=False):
        if dir_opt:
            # Optimize direction. Note that the optimization velocity is of unit length in this case.
            self.new_velocity = vel_pref * maxSpeed
        elif absSq(vel_pref) > sqr(maxSpeed):
            # Optimize closest point and outside circle.
            self.new_velocity = (vel_pref / np.linalg.norm(vel_pref)) * maxSpeed
        else:
            # Optimize closest point and inside circle.
            self.new_velocity = vel_pref

        for i in range(len(planes)):
            if np.dot(planes[i].normal, (planes[i].point - self.new_velocity)) > 0.0:   # new_velocity不在ORCA内部
                # Result does not satisfy constraint i. Compute new optimal result.
                tempResult = self.new_velocity

                if not self.linearProgram2(planes, i, maxSpeed, vel_pref, dir_opt):
                    self.new_velocity = tempResult
                    return i    # 没有找到最优解，在find_next_action里面用线性规划linearProgram4找

        return len(planes)      # 满足约束条件，找到最优解

    def linearProgram4(self, planes, beginPlane, radius):

        for i in range(beginPlane, len(planes)):
            if np.dot(planes[i].normal, (planes[i].point - self.new_velocity) > 0.0):
                # Result does not satisfy constraint of plane i.
                projPlanes = []

                for j in range(i):
                    plane = Plane()

                    crossProduct = np.cross(planes[j].normal, planes[i].normal)

                    if absSq(crossProduct) <= rvo3d_epsilon:
                        # Plane i and plane j are (almost) parallel.
                        if np.dot(planes[i].normal, planes[j].normal) > 0.0:
                            # Plane i and plane j point in the same direction.
                            continue
                        else:
                            # Plane i and plane j point in opposite direction.
                            plane.point = 0.5 * (planes[i].point + planes[j].point)

                    else:
                        # Plane.point is point on line of intersection between plane i and plane j.
                        lineNormal = np.cross(crossProduct, planes[i].normal)
                        dot_product1 = np.dot((planes[j].point - planes[i].point), planes[j].normal)
                        dot_product2 = np.dot(lineNormal, planes[j].normal)
                        plane.point = planes[i].point + (dot_product1 / dot_product2) * lineNormal

                    plane.normal = normalize(planes[j].normal - planes[i].normal)
                    projPlanes.append(plane)

                tempResult = self.new_velocity

                if self.linearProgram3(projPlanes, radius, planes[i].normal, dir_opt=True) < len(projPlanes):
                    '''This should in principle not happen.The result is by definition already 
                        in the feasible region of this linear program.If it fails, it is due to small 
                        floating point error, and the current result is kept.'''
                    self.new_velocity = tempResult

                dist = np.dot(planes[i].normal, (planes[i].point - self.new_velocity))

    def get_trajectory(self, agent):  # global planning added A* planner
        if agent.path:
            if self.now_goal is None:  # first
                self.now_goal = np.array(agent.path.pop(), dtype='float64')
            dis = distance(agent.pos_global_frame, self.now_goal)
            dis_nowgoal_globalgoal = distance(self.now_goal, agent.goal_global_frame)
            dis_nowgoal_globalpos = distance(agent.pos_global_frame, agent.goal_global_frame)
            if dis <= self.update_nowGoal_dist * agent.radius:  # free collision
                if agent.path:
                    self.now_goal = np.array(agent.path.pop(), dtype='float64')
            elif dis_nowgoal_globalgoal >= dis_nowgoal_globalpos:  # free back to current now_goal when free collision
                if agent.path:
                    self.now_goal = np.array(agent.path.pop(), dtype='float64')
        else:
            self.now_goal = agent.goal_global_frame


def computeNeighbors(agent, kdTree):
    if agent.is_collision:
        return

    agent.neighbors.clear()
    rangeSq = agent.neighborDist ** 2
    # check obstacle neighbors
    kdTree.computeObstacleNeighbors(agent, rangeSq)
    # check other agents
    kdTree.computeAgentNeighbors(agent, rangeSq)


def cartesian2spherical(agent, vA_post):
    speed = distance(vA_post, [0, 0, 0])
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


def compute_v_pref(goal, agent):
    if agent.desire_path_length is None:
        agent.desire_path_length = distance(agent.pos_global_frame, agent.goal_global_frame) - 0.5
        agent.desire_points_num = agent.desire_path_length / DT
    dif_x = goal - agent.pos_global_frame
    norm = int(distance(dif_x, [0, 0, 0]) * eps) / eps
    norm_dif_x = dif_x * agent.pref_speed / norm
    v_pref = np.array(norm_dif_x)
    if reached(agent.goal_global_frame, agent.pos_global_frame, bound=0.2):
        v_pref[0] = 0.0
        v_pref[1] = 0.0
        v_pref[2] = 0.0
    agent.v_pref = v_pref
    V_des = np.array([int(v_pref[0] * eps) / eps, int(v_pref[1] * eps) / eps, int(v_pref[2] * eps) / eps])
    return V_des


if __name__ == "__main__":
    pass
