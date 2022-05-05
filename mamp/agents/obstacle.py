import numpy as np
from math import sqrt


class Obstacle(object):
    def __init__(self, pos, shape_dict, id):
        self.shape = shape = shape_dict['shape']
        self.feature = feature = shape_dict['feature']
        if shape == 'cube':
            self.length, self.width, self.height = shape_dict['feature']
            self.radius = sqrt(self.length ** 2 + self.width ** 2 + self.height ** 2) / 2
        elif shape == 'sphere':
            self.radius = shape_dict['feature']
        else:
            raise NotImplementedError
        self.pos_global_frame = np.array(pos, dtype='float64')
        self.vel_global_frame = np.array([0.0, 0.0, 0.0])
        self.pos = pos
        self.id = id
        self.t = 0.0
        self.step_num = 0
        self.is_at_goal = True
        self.is_obstacle = True
        self.was_in_collision_already = False
        self.is_collision = False

        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]