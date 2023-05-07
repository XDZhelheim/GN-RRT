"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random

import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import time


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
        start,
        goal,
        obstacle_list,
        rand_area,
        expand_dis=6,
        path_resolution=2,
        goal_sample_rate=5,
        max_iter=500,
        play_area=None,
    ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,x_len,y_len],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
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

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()  # (int, int)
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
            if delta_x > 1 or delta_y > 1:
                new_node.path_x.append(int(new_node.x))
                new_node.path_y.append(int(new_node.y))

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

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.randint(self.min_rand, self.max_rand),
                random.randint(self.min_rand, self.max_rand),
            )
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
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
            plt.fill(
                [obs_x, obs_x + obs_x_len, obs_x + obs_x_len, obs_x],
                [obs_y, obs_y, obs_y + obs_y_len, obs_y + obs_y_len],
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


def main(gx, gy):
    print("start " + __file__)

    # 所有障碍物都是长方形 [x, y, x_len, y_len] xy是长方形左下角坐标
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
    print("time: {:.2f} s".format(e - s))
    if path is None:
        print("Cannot find path")
    else:
        print(np.array(path))

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], "-r")
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == "__main__":
    DEBUG = False
    show_animation = True
    main(gx=100, gy=100)
