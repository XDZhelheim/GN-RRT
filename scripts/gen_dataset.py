import numpy as np

# x: (n*p, num_grid_w, num_grid_h, channels)
# y: (n*p, num_grid_w, num_grid_h) 在路径的点占整个grid里点的比例+在路径的点占整个路径里点的比例


def gen_grid_xy(images, paths, num_grid_w=20, num_grid_h=20):
    """
    images: (n, w, h)
    paths: (n, num_paths, path_len(not const), 2) !!this is not a matrix

    will generate n*p samples, n=num_images, p=num_paths

    channels:
    - 障碍物占 grid 比例 (float)
    - 是否为起/终所在的格子 (0 or 1)
    """
    num_images = images.shape[0]
    num_paths = len(paths[0])

    x = np.zeros((num_images, num_paths, num_grid_w, num_grid_h, 2))
    y = np.zeros((num_images, num_paths, num_grid_w, num_grid_h))
    for i in range(num_images):
        image = images[i]

        # 统计每个格子障碍物点数占比
        for idxx in range(image.shape[0]):
            for idxy in range(image.shape[1]):
                if image[idxx, idxy] == 1:
                    x[i, :, idxx // num_grid_w, idxy // num_grid_h, 0] += 1
        x[i, :, :, :, 0] /= num_grid_w * num_grid_h

        path_list = paths[i]
        for j in range(len(path_list)):
            path = path_list[j]  # (path_len, 2)
            # 标记起点格 终点格
            x[i, j, path[0][0] // num_grid_w, path[0][1] // num_grid_h, 1] = 1
            x[i, j, path[-1][0] // num_grid_w, path[-1][1] // num_grid_h, 1] = 1
            
            for point in path:
                y[i, j, point[0] // num_grid_w, point[1] // num_grid_h] += 1
            # 计算 y
            y[i, j] = y[i, j] / (num_grid_w * num_grid_h) + y[i, j] / len(path)

    print("x:", x.shape)
    print("y:", y.shape)

    x = x.reshape(num_images * num_paths, num_grid_w, num_grid_h, 2)
    y = y.reshape(num_images * num_paths, num_grid_w, num_grid_h)

    return x, y


def gen_grid_dataset(n, p, num_grid_w=20, num_grid_h=20, train_size=0.7, val_size=0.1):
    images = np.load(f"../data/n_{n}_p_{p}/image_{n}_{p}.npz")["data"]
    paths = np.load(f"../data/n_{n}_p_{p}/path_{n}_{p}.npz", allow_pickle=True)["data"]

    x, y = gen_grid_xy(images, paths, num_grid_w, num_grid_h)

    num_samples = x.shape[0]
    split1 = int(num_samples * train_size)
    split2 = int(num_samples * (train_size + val_size))

    x_train = x[:split1]
    x_val = x[split1:split2]
    x_test = x[split2:]

    y_train = y[:split1]
    y_val = y[split1:split2]
    y_test = y[split2:]

    print("x_train:", x_train.shape)
    print("x_val:", x_val.shape)
    print("x_test:", x_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    np.savez_compressed(
        f"../data/n_{n}_p_{p}/train_{n}_{p}.npz", x_train=x_train, y_train=y_train
    )
    np.savez_compressed(
        f"../data/n_{n}_p_{p}/val_{n}_{p}.npz", x_val=x_val, y_val=y_val
    )
    np.savez_compressed(
        f"../data/n_{n}_p_{p}/test_{n}_{p}.npz", x_test=x_test, y_test=y_test
    )
    
if __name__ == "__main__":
    gen_grid_dataset(n=20, p=10)
