import numpy as np

# x: (n*p, grid_width, grid_height, channels)
# y: (n*p, grid_width, grid_height) 在路径的点占整个grid里点的比例


def gen_grid_dataset(images, paths, n_grid_w=10, n_grid_h=10):
    """
    images: (n, h, w)
    paths: (n, num_paths, path_len(not const), 2) !!this is not a matrix

    will generate n*p samples, n=num_images, p=num_paths

    channels:
    - 障碍物占 grid 比例 (float)
    - 是否为起/终所在的格子 (0 or 1)
    """
    num_images, image_width, image_height = images.shape
    num_paths = len(paths[0])
    grid_width = image_width / n_grid_w
    grid_height = image_height / n_grid_h

    x = np.zeros(num_images*num_paths, grid_width, grid_height, 2)
    y = np.zeros(num_images*num_paths, grid_width, grid_height)
    for i in range(num_images):
        pass
