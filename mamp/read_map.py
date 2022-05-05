import numpy as np
from mamp.agents.obstacle import Obstacle


class Voxels(object):
    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order


def read_as_3d_array(fp, fix_coords=True):
    dims, translate, scale = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return Voxels(data, dims, translate, scale, axis_order)


def read_header(fp):
    # Read binvox head file
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale


def read_obstacle(center, environ, obs_path):
    obstacles = []
    if environ == "exp3":
        resolution = 0.1
        bias = [-13.5, -13.5, -1.4]
        # path = "./map.binvox"
    else:
        return obstacles

    with open(obs_path, 'rb') as f:
        model = read_as_3d_array(f)
        print('size(l, w, h):', model.dims, 'translate matrix:', model.translate, 'scale factor:', model.scale)
    count = 0
    floor_count = 0
    tree_count = 0
    for x in range(model.dims[0]):
        for y in range(model.dims[2]):
            for z in range(model.dims[1]):
                if model.data[x][y][z]:
                    position = [(y + model.translate[1]) * resolution + bias[0] + center[0],
                                (x + model.translate[0]) * resolution + bias[1] + center[1],
                                z * resolution + bias[2]]
                    if position[2] > -1:
                        if tree_count == 10:  # Display the obstacles above ground
                            Obs = Obstacle(pos=position, shape_dict={'shape': "sphere", 'feature': 0.2},
                                           id=count)
                            obstacles.append(Obs)
                            count += 1
                            tree_count = 0
                            # print(count, ":", position)
                        else:
                            tree_count += 1
                    else:
                        if floor_count == 1000:  # Not display the ground
                            Obs = Obstacle(pos=position, shape_dict={'shape': "sphere", 'feature': 0.2},
                                           id=count)
                            obstacles.append(Obs)
                            count += 1
                            # print(count, ":", position)
                            floor_count = 0
                        else:
                            floor_count += 1
    print("obstacle_num:", count)
    return obstacles