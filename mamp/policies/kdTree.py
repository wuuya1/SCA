"""
@ Author: Gang Xu
@ Date: 2022.04.16
@ Details: KDTree 3D
@ Reference: Reciprocal n-Body Collision Avoidance*
@ Github: https://github.com/snape/RVO2-3D
"""

import numpy as np
from mamp.util import sqr


class AgentTreeNode(object):
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.left = 0
        self.right = 0
        self.maxCoord = np.array([0.0, 0.0, 0.0])
        self.minCoord = np.array([0.0, 0.0, 0.0])


class ObstacleTreeNode(object):
    def __init__(self):
        self.begin = 0
        self.end = 0
        self.left = 0
        self.right = 0
        self.maxCoord = np.array([0.0, 0.0, 0.0])
        self.minCoord = np.array([0.0, 0.0, 0.0])


class KDTree(object):
    def __init__(self, agents, obstacles):
        self.agents = agents
        self.obstacles = obstacles
        self.agent_tree_node = AgentTreeNode()
        self.obstacle_tree_node = ObstacleTreeNode()
        self.agentTree = []
        for i in range(2 * len(agents) - 1):
            self.agentTree.append(AgentTreeNode())
        self.agentIDs = []
        for obj in agents:
            self.agentIDs.append(obj.id)

        self.obstacleTree = []
        for i in range(2 * len(obstacles) - 1):
            self.obstacleTree.append(ObstacleTreeNode())
        self.obstacleIDs = []
        for obj in obstacles:
            self.obstacleIDs.append(obj.id)
        self.max_leaf_size = 10
        self.epsilon = 1e-5

    def buildAgentTree(self):
        if self.agentIDs:
            self.buildAgentTreeRecursive(0, len(self.agentIDs), 0)

    def buildAgentTreeRecursive(self, begin, end, node):
        self.agentTree[node].begin = begin
        self.agentTree[node].end = end
        # 不直接用数组进行赋值是因为会指向同一地址，智能体的当前位置也会跟随minCoord, maxCoord改变
        self.agentTree[node].minCoord[0] = self.agents[self.agentIDs[begin]].pos_global_frame[0]
        self.agentTree[node].minCoord[1] = self.agents[self.agentIDs[begin]].pos_global_frame[1]
        self.agentTree[node].minCoord[2] = self.agents[self.agentIDs[begin]].pos_global_frame[2]
        self.agentTree[node].maxCoord[0] = self.agents[self.agentIDs[begin]].pos_global_frame[0]
        self.agentTree[node].maxCoord[1] = self.agents[self.agentIDs[begin]].pos_global_frame[1]
        self.agentTree[node].maxCoord[2] = self.agents[self.agentIDs[begin]].pos_global_frame[2]

        for i in range(begin + 1, end):
            self.agentTree[node].maxCoord[0] = max(self.agentTree[node].maxCoord[0],
                                                   self.agents[self.agentIDs[i]].pos_global_frame[0])
            self.agentTree[node].minCoord[0] = min(self.agentTree[node].minCoord[0],
                                                   self.agents[self.agentIDs[i]].pos_global_frame[0])
            self.agentTree[node].maxCoord[1] = max(self.agentTree[node].maxCoord[1],
                                                   self.agents[self.agentIDs[i]].pos_global_frame[1])
            self.agentTree[node].minCoord[1] = min(self.agentTree[node].minCoord[1],
                                                   self.agents[self.agentIDs[i]].pos_global_frame[1])
            self.agentTree[node].maxCoord[2] = max(self.agentTree[node].maxCoord[2],
                                                   self.agents[self.agentIDs[i]].pos_global_frame[2])
            self.agentTree[node].minCoord[2] = min(self.agentTree[node].minCoord[2],
                                                   self.agents[self.agentIDs[i]].pos_global_frame[2])

        if end - begin > self.max_leaf_size:  # No leaf node
            dif0 = self.agentTree[node].maxCoord[0] - self.agentTree[node].minCoord[0]
            dif1 = self.agentTree[node].maxCoord[1] - self.agentTree[node].minCoord[1]
            dif2 = self.agentTree[node].maxCoord[2] - self.agentTree[node].minCoord[2]
            if dif0 > dif1 and dif0 > dif2:
                coord = 0
            elif dif1 > dif2:
                coord = 1
            else:
                coord = 2

            splitValue = 0.5 * (self.agentTree[node].maxCoord[coord] + self.agentTree[node].minCoord[coord])

            left = begin
            right = end

            while left < right:
                while left < right and self.agents[self.agentIDs[left]].pos_global_frame[coord] < splitValue:
                    left += 1

                while right > left and self.agents[self.agentIDs[right - 1]].pos_global_frame[coord] >= splitValue:
                    right -= 1

                if left < right:
                    self.agentIDs[left], self.agentIDs[right - 1] = self.agentIDs[right - 1], self.agentIDs[left]
                    left += 1
                    right -= 1
            leftSize = left - begin
            if leftSize == 0:
                leftSize += 1
                left += 1
                right += 1

            self.agentTree[node].left = node + 1
            self.agentTree[node].right = node + 2 * leftSize  # node + 1 + (2 * leftsize - 1)

            self.buildAgentTreeRecursive(begin, left, self.agentTree[node].left)
            self.buildAgentTreeRecursive(left, end, self.agentTree[node].right)

    def computeAgentNeighbors(self, agent, rangeSq):
        self.queryAgentTreeRecursive(agent, rangeSq, 0)

    def queryAgentTreeRecursive(self, agent, rangeSq, node):
        if self.agentTree[node].end - self.agentTree[node].begin <= self.max_leaf_size:
            for i in range(self.agentTree[node].begin, self.agentTree[node].end):
                agent.insertAgentNeighbor(self.agents[self.agentIDs[i]], rangeSq)
        else:
            distSqLeft = (
                    sqr(max(0.0, self.agentTree[self.agentTree[node].left].minCoord[0] - agent.pos_global_frame[0])) +
                    sqr(max(0.0, agent.pos_global_frame[0] - self.agentTree[self.agentTree[node].left].maxCoord[0])) +
                    sqr(max(0.0, self.agentTree[self.agentTree[node].left].minCoord[1] - agent.pos_global_frame[1])) +
                    sqr(max(0.0, agent.pos_global_frame[1] - self.agentTree[self.agentTree[node].left].maxCoord[1])) +
                    sqr(max(0.0, self.agentTree[self.agentTree[node].left].minCoord[2] - agent.pos_global_frame[2])) +
                    sqr(max(0.0, agent.pos_global_frame[2] - self.agentTree[self.agentTree[node].left].maxCoord[2])))
            distSqRight = (
                    sqr(max(0.0, self.agentTree[self.agentTree[node].right].minCoord[0] - agent.pos_global_frame[0])) +
                    sqr(max(0.0, agent.pos_global_frame[0] - self.agentTree[self.agentTree[node].right].maxCoord[0])) +
                    sqr(max(0.0, self.agentTree[self.agentTree[node].right].minCoord[1] - agent.pos_global_frame[1])) +
                    sqr(max(0.0, agent.pos_global_frame[1] - self.agentTree[self.agentTree[node].right].maxCoord[1])) +
                    sqr(max(0.0, self.agentTree[self.agentTree[node].right].minCoord[2] - agent.pos_global_frame[2])) +
                    sqr(max(0.0, agent.pos_global_frame[2] - self.agentTree[self.agentTree[node].right].maxCoord[2])))

            if distSqLeft < distSqRight:
                if distSqLeft < rangeSq:
                    self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].left)
                    if distSqRight < rangeSq:
                        self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].right)
            else:
                if distSqRight < rangeSq:
                    self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].right)
                    if distSqLeft < rangeSq:
                        self.queryAgentTreeRecursive(agent, rangeSq, self.agentTree[node].left)

    def buildObstacleTree(self):
        if self.obstacleIDs:
            self.buildObstacleTreeRecursive(0, len(self.obstacleIDs), 0)

    def buildObstacleTreeRecursive(self, begin, end, node):
        self.obstacleTree[node].begin = begin
        self.obstacleTree[node].end = end

        self.obstacleTree[node].minCoord[0] = self.obstacles[self.obstacleIDs[begin]].pos_global_frame[0]
        self.obstacleTree[node].minCoord[1] = self.obstacles[self.obstacleIDs[begin]].pos_global_frame[1]
        self.obstacleTree[node].minCoord[2] = self.obstacles[self.obstacleIDs[begin]].pos_global_frame[2]
        self.obstacleTree[node].maxCoord[0] = self.obstacles[self.obstacleIDs[begin]].pos_global_frame[0]
        self.obstacleTree[node].maxCoord[1] = self.obstacles[self.obstacleIDs[begin]].pos_global_frame[1]
        self.obstacleTree[node].maxCoord[2] = self.obstacles[self.obstacleIDs[begin]].pos_global_frame[2]

        for i in range(begin + 1, end):
            self.obstacleTree[node].maxCoord[0] = max(self.obstacleTree[node].maxCoord[0],
                                                      self.obstacles[self.obstacleIDs[i]].pos_global_frame[0])
            self.obstacleTree[node].minCoord[0] = min(self.obstacleTree[node].minCoord[0],
                                                      self.obstacles[self.obstacleIDs[i]].pos_global_frame[0])
            self.obstacleTree[node].maxCoord[1] = max(self.obstacleTree[node].maxCoord[1],
                                                      self.obstacles[self.obstacleIDs[i]].pos_global_frame[1])
            self.obstacleTree[node].minCoord[1] = min(self.obstacleTree[node].minCoord[1],
                                                      self.obstacles[self.obstacleIDs[i]].pos_global_frame[1])
            self.obstacleTree[node].maxCoord[2] = max(self.obstacleTree[node].maxCoord[2],
                                                      self.obstacles[self.obstacleIDs[i]].pos_global_frame[2])
            self.obstacleTree[node].minCoord[2] = min(self.obstacleTree[node].minCoord[2],
                                                      self.obstacles[self.obstacleIDs[i]].pos_global_frame[2])

        if end - begin > self.max_leaf_size:  # No leaf node
            dif0 = self.obstacleTree[node].maxCoord[0] - self.obstacleTree[node].minCoord[0]
            dif1 = self.obstacleTree[node].maxCoord[1] - self.obstacleTree[node].minCoord[1]
            dif2 = self.obstacleTree[node].maxCoord[2] - self.obstacleTree[node].minCoord[2]
            if dif0 > dif1 and dif0 > dif2:
                coord = 0
            elif dif1 > dif2:
                coord = 1
            else:
                coord = 2

            splitValue = 0.5 * (self.obstacleTree[node].maxCoord[coord] + self.obstacleTree[node].minCoord[coord])

            left = begin
            right = end

            while left < right:
                while left < right and self.obstacles[self.obstacleIDs[left]].pos_global_frame[coord] < splitValue:
                    left += 1

                while right > left and self.obstacles[self.obstacleIDs[right - 1]].pos_global_frame[
                      coord] >= splitValue:
                    right -= 1

                if left < right:
                    self.obstacleIDs[left], self.obstacleIDs[right - 1] = self.obstacleIDs[right - 1], self.obstacleIDs[
                        left]
                    left += 1
                    right -= 1
            leftSize = left - begin
            if leftSize == 0:
                leftSize += 1
                left += 1
                right += 1

            self.obstacleTree[node].left = node + 1
            self.obstacleTree[node].right = node + 2 * leftSize  # node + 1 + (2 * leftsize - 1)

            self.buildObstacleTreeRecursive(begin, left, self.obstacleTree[node].left)
            self.buildObstacleTreeRecursive(left, end, self.obstacleTree[node].right)

    def computeObstacleNeighbors(self, agent, rangeSq):
        self.queryObstacleTreeRecursive(agent, rangeSq, 0)

    def queryObstacleTreeRecursive(self, agent, rangeSq, node):
        if self.obstacleTree:
            if self.obstacleTree[node].end - self.obstacleTree[node].begin <= self.max_leaf_size:
                for i in range(self.obstacleTree[node].begin, self.obstacleTree[node].end):
                    agent.insertObstacleNeighbor(self.obstacles[self.obstacleIDs[i]], rangeSq)
            else:
                distSqLeft = (
                 sqr(max(0.0, self.obstacleTree[self.obstacleTree[node].left].minCoord[0] - agent.pos_global_frame[0])) +
                 sqr(max(0.0, agent.pos_global_frame[0] - self.obstacleTree[self.obstacleTree[node].left].maxCoord[0])) +
                 sqr(max(0.0, self.obstacleTree[self.obstacleTree[node].left].minCoord[1] - agent.pos_global_frame[1])) +
                 sqr(max(0.0, agent.pos_global_frame[1] - self.obstacleTree[self.obstacleTree[node].left].maxCoord[1])) +
                 sqr(max(0.0, self.obstacleTree[self.obstacleTree[node].left].minCoord[2] - agent.pos_global_frame[2])) +
                 sqr(max(0.0, agent.pos_global_frame[2] - self.obstacleTree[self.obstacleTree[node].left].maxCoord[2])))
                distSqRight = (
                 sqr(max(0.0, self.obstacleTree[self.obstacleTree[node].right].minCoord[0] - agent.pos_global_frame[0])) +
                 sqr(max(0.0, agent.pos_global_frame[0] - self.obstacleTree[self.obstacleTree[node].right].maxCoord[0])) +
                 sqr(max(0.0, self.obstacleTree[self.obstacleTree[node].right].minCoord[1] - agent.pos_global_frame[1])) +
                 sqr(max(0.0, agent.pos_global_frame[1] - self.obstacleTree[self.obstacleTree[node].right].maxCoord[1])) +
                 sqr(max(0.0, self.obstacleTree[self.obstacleTree[node].right].minCoord[2] - agent.pos_global_frame[2])) +
                 sqr(max(0.0, agent.pos_global_frame[2] - self.obstacleTree[self.obstacleTree[node].right].maxCoord[2])))

                if distSqLeft < distSqRight:
                    if distSqLeft < rangeSq:
                        self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].left)
                        if distSqRight < rangeSq:
                            self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].right)
                else:
                    if distSqRight < rangeSq:
                        self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].right)
                        if distSqLeft < rangeSq:
                            self.queryObstacleTreeRecursive(agent, rangeSq, self.obstacleTree[node].left)
