"""

Grid RRT & NeuralGridRRT

Author: Zheng Dong
Adapted from rrt.py by AtsushiSakai(@Atsushi_twi)

"""

import math
import random

# add these two lines on Windows
# import matplotlib
# matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import os
import sys

sys.path.append("..")
from lib.utils import seed_everything
from gen_dataset import gen_grid_xy
from models.GridGCN import GridGCN


def softmax(x):
    temp = np.exp(x - np.max(x))
    f_x = temp / temp.sum()
    return f_x


class GridRRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

        def __str__(self):
            return str((self.x, self.y))

    class AreaBounds:
        def __init__(self, area):
            self.xmin = area[0]
            self.xmax = area[1]
            self.ymin = area[2]
            self.ymax = area[3]

    def __init__(
        self,
        start=None,
        goal=None,
        obstacle_list=None,
        rand_area=None,
        expand_dis=5,
        path_resolution=1,
        goal_sample_rate=0,
        max_iter=500,
        play_area=None,
        model_pred=[],
    ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,x_len,y_len],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]

        """
        if start and goal:
            self.set_start_end(start, goal)

        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

        self.model_pred = model_pred
        if len(model_pred) > 0:
            self.nh, self.nw = model_pred.shape
            self.gh, self.gw = (
                self.play_area.xmax // self.nh,
                self.play_area.ymax // self.nw,
            )

    def set_start_end(self, start, end):
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(end[0], end[1])

    def set_model_pred(self, model_pred):
        self.model_pred = model_pred
        if len(model_pred) > 0:
            self.nh, self.nw = model_pred.shape
            self.gh, self.gw = (
                self.play_area.xmax // self.nh,
                self.play_area.ymax // self.nw,
            )
            self.in_same_grid = (
                self.start.x // self.gh == self.end.x // self.gh
                and self.start.y // self.gw == self.end.y // self.gw
            )

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node(
                cur_node=self.node_list[
                    self.get_nearest_node_index(self.node_list, self.end)
                ]
            )  # (int, int)
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            if DEBUG:
                print(rnd_node, nearest_ind, nearest_node)

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if DEBUG:
                print(new_node)

            if self.check_if_outside_play_area(
                new_node, self.play_area
            ) and self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if DEBUG:
                for node in self.node_list:
                    print(node, end=" ")
                print()

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if DEBUG:
                print(
                    self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y)
                )

            if (
                self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y)
                <= self.expand_dis
            ):
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if DEBUG:
                    print(final_node)
                if self.check_collision(final_node, self.obstacle_list):
                    if DEBUG:
                        print(final_node.path_x, final_node.path_y)
                    return self.generate_final_course(len(self.node_list) - 1), i

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None, i  # cannot find path

    def steer(self, from_node, to_node, extend_length=99999):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        # ! 保证 expand 的每个 node 坐标都是 int，同时内部累加的时候依然是 float
        for _ in range(n_expand):
            delta_x = self.path_resolution * math.cos(theta)
            delta_y = self.path_resolution * math.sin(theta)
            new_node.x += delta_x
            new_node.y += delta_y
            # ! path_x path_y 貌似只在画图时用了 不影响最终结果路径
            # ! 如果在这里强转 int 则画出来的树有断裂
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y
        else:
            new_node.x = int(new_node.x)
            new_node.y = int(new_node.y)

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self, cur_node: Node):
        # ! use model_pred to guide sampling
        # cur_node 是当前树上离终点最近的点
        if len(self.model_pred) > 0:
            return self.get_random_node_by_model(cur_node)

        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.randint(self.min_rand, self.max_rand),
                random.randint(self.min_rand, self.max_rand),
            )
        else:  # goal point sampling
            # ! 这里 goal_sample_rate 设为 0, 取消他的 heuristic
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def get_random_node_by_model(self, cur_node: Node):
        cur_grid_x = cur_node.x // self.gh
        cur_grid_y = cur_node.y // self.gw

        # 以当前点的九宫格
        prob = -np.ones((3, 3)) * np.inf
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (
                    cur_grid_x + i >= 0
                    and cur_grid_x + i < self.nh
                    and cur_grid_y + j >= 0
                    and cur_grid_y + j < self.nw
                ):
                    prob[i + 1, j + 1] = self.model_pred[cur_grid_x + i, cur_grid_y + j]
        if not self.in_same_grid:
            prob[1, 1] = -np.inf  # 不让你选自己的格子
        # 如果起点终点在同格子 则自己的格子也进入候选

        prob = softmax(prob)

        # 按预测值加权随机挑选一个格子
        rand_number = random.random()
        prob_sum = 0
        found = False
        for i in range(3):
            for j in range(3):
                prob_sum += prob[i, j]
                if prob_sum > rand_number:
                    found = True
                    delta_x = i
                    delta_y = j
                    break
            if found:
                break

        # 坐标转换
        delta_x -= 1
        delta_y -= 1

        chosen_grid_x = cur_grid_x + delta_x
        chosen_grid_y = cur_grid_y + delta_y

        assert chosen_grid_x >= 0 and chosen_grid_x < self.nh, chosen_grid_x
        assert chosen_grid_y >= 0 and chosen_grid_y < self.nw, chosen_grid_y

        rand_x_min = chosen_grid_x * self.gh
        rand_x_max = rand_x_min + self.gh
        rand_y_min = chosen_grid_y * self.gw
        rand_y_max = rand_y_min + self.gw

        rnd = self.Node(
            random.randint(rand_x_min, rand_x_max),
            random.randint(rand_y_min, rand_y_max),
        )

        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for obs_x, obs_y, obs_x_len, obs_y_len in self.obstacle_list:
            xmax = (
                obs_x + obs_x_len
                if obs_x + obs_x_len < self.play_area.xmax
                else self.play_area.xmax
            )
            ymax = (
                obs_y + obs_y_len
                if obs_y + obs_y_len < self.play_area.ymax
                else self.play_area.ymax
            )
            plt.fill(
                [obs_x, xmax, xmax, obs_x],
                [obs_y, obs_y, ymax, ymax],
                "black",
            )

        if self.play_area is not None:
            plt.plot(
                [
                    self.play_area.xmin,
                    self.play_area.xmax,
                    self.play_area.xmax,
                    self.play_area.xmin,
                    self.play_area.xmin,
                ],
                [
                    self.play_area.ymin,
                    self.play_area.ymin,
                    self.play_area.ymax,
                    self.play_area.ymax,
                    self.play_area.ymin,
                ],
                "-k",
            )

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        # plt.axis([0, 200, 0, 200])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [
            (node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
            for node in node_list
        ]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):
        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if (
            node.x < play_area.xmin
            or node.x > play_area.xmax
            or node.y < play_area.ymin
            or node.y > play_area.ymax
        ):
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList):
        if node is None:
            return False

        for obs_x, obs_y, obs_x_len, obs_y_len in obstacleList:
            for node_x, node_y in zip(node.path_x, node.path_y):
                if (
                    node_x >= obs_x
                    and node_y >= obs_y
                    and obs_x + obs_x_len > node_x
                    and obs_y + obs_y_len > node_y
                ):
                    return False

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def rrt(gx, gy):
    print("start " + __file__)

    # 所有障碍物都是长方形 [x, y, x_len, y_len] xy是长方形起始点坐标
    obstacleList = [
        [10, 10, 10, 10],
        [10, 20, 5, 5],
        [10, 0, 20, 20],
    ]
    # Set Initial parameters
    rrt = GridRRT(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[0, 200],
        obstacle_list=obstacleList,
        play_area=[0, 200, 0, 200],
        max_iter=200,
    )
    # 200*200 image, 整个代码里所有node坐标都是整数

    s = time.time()
    path, num_iter = rrt.planning(animation=show_animation)
    e = time.time()

    print(f"num_iter = {num_iter}")
    print("time: {:.8f} s".format(e - s))
    if path is None:
        print("Cannot find path")
    else:
        print(np.array(path))

        # Draw final path
        if draw_final:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], "-r")
            plt.xticks(np.arange(0, 200 + 20, step=20))
            plt.yticks(np.arange(0, 200 + 20, step=20))
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.savefig("../images/temp_grid_rrt.png", dpi=300, bbox_inches="tight")
            plt.show()


@torch.no_grad()
def test_image_dataset(
    n=300,
    p=20,
    h=200,
    w=200,
    nh=20,
    nw=20,
    model=None,
    batch_size=32,
    max_iter=400,
    draw_list=None,
):
    if draw_list:
        image_path = f"../images/n_{n}_p_{p}"
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if model:
            image_path = os.path.join(image_path, "nrrt")
            if not os.path.exists(image_path):
                os.makedirs(image_path)
        else:
            image_path = os.path.join(image_path, "rrt")
            if not os.path.exists(image_path):
                os.makedirs(image_path)

    # images: (n, h, w)
    # obss: (n, obs_number, 4)
    # paths: (n, num_paths, path_len(not const), 2)
    images = np.load(f"../data/n_{n}_p_{p}/image_{n}_{p}.npz")["data"]
    paths = np.load(f"../data/n_{n}_p_{p}/path_{n}_{p}.npz", allow_pickle=True)["data"]
    obss = np.load(f"../data/n_{n}_p_{p}/obs_{n}_{p}.npz")["data"]

    if model:
        x_test, _ = gen_grid_xy(
            images, paths, num_grid_h=nh, num_grid_w=nw
        )  # (n*p, nh, nw, num_channels)
        model = model.to(DEVICE)
        model.eval()

        testset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_test), torch.zeros(size=x_test.shape)
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False
        )

        out = []
        for x_batch, _ in testloader:
            x_batch = x_batch.to(DEVICE)
            out_batch = model(x_batch)

            out_batch = out_batch.cpu().numpy()
            out.append(out_batch)
        pred = np.vstack(out).squeeze()  # (n*p, nh, nw)
        pred = pred.reshape(n, p, nh, nw)

        print("----- Grid Neural RRT -----")
    else:
        print("----- Naive RRT -----")

    rrt = GridRRT(
        rand_area=[0, max(h, w)],
        play_area=[0, h, 0, w],
        max_iter=max_iter,
    )

    time_all = 0
    iter_all = 0
    succ_all = 0
    for idx_n in range(n):
        obs_list = obss[idx_n]
        rrt.obstacle_list = obs_list
        for idx_p in range(p):
            a_star_path = paths[idx_n, idx_p]
            start_point = a_star_path[0]
            end_point = a_star_path[-1]

            rrt.set_start_end(start_point, end_point)

            if model:
                rrt.set_model_pred(pred[idx_n, idx_p])  # (nh, nw)

            s = time.time()
            path, num_iter = rrt.planning(animation=False)
            e = time.time()

            time_all += e - s
            iter_all += num_iter
            if path:
                succ_all += 1

            if [idx_n, idx_p] in draw_list and path:
                print(f"Draw n={idx_n} p={idx_p}")
                rrt.draw_graph()
                plt.plot([x for (x, y) in path], [y for (x, y) in path], "-r")
                plt.xticks(np.arange(0, h + nh, step=nh))
                plt.yticks(np.arange(0, w + nw, step=nw))
                plt.grid(True)
                plt.savefig(
                    os.path.join(image_path, f"{idx_n}_{idx_p}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )

    time_avg = time_all / (n * p)
    iter_avg = iter_all / (n * p)
    succ_rate = succ_all / (n * p)

    print("Avg time:", time_avg)
    print("Avg iters:", iter_avg)
    print("Success rate:", succ_rate)


if __name__ == "__main__":
    seed_everything(233)
    DEBUG = False
    show_animation = False
    draw_final = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model_param_path = (
    #     "../saved_models/GridGCN/GridGCN-n_2500_p_20-2023-05-09-23-29-49.pt"
    # )
    model_param_path = (
        "../saved_models/GridGCN/GridGCN-n_500_p_20-2023-05-09-19-26-12.pt"
    )

    model = GridGCN(device=DEVICE)
    model.load_state_dict(torch.load(model_param_path))
    model = model.to(DEVICE)

    n = 20
    p = 10

    draw_list = []
    draw_list += [[i, 0] for i in range(20)]
    draw_list += np.arange(0, 20).reshape(-1, 2).tolist()

    # model=None: use naive RRT
    # model not None: use GridNeuralRRT
    test_image_dataset(n=n, p=p, draw_list=draw_list, model=model)
