"""

Path planning Sample Code with RRT with Reeds-Shepp path

author: AtsushiSakai(@Atsushi_twi)

"""
import copy
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

#sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                #"/../ReedsSheppPath/")
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                #"/../RRTStar/")


import reeds_shepp_path_planning
from rrt_star import RRTStar

show_animation = True


class RRTStarReedsShepp(RRTStar):
    """
    Class for RRT star planning with Reeds Shepp path
    """

    class Node(RRTStar.Node):
        """
        RRT Node
        """

        def __init__(self, x, y, yaw):
            super().__init__(x, y)
            self.yaw = yaw
            self.path_yaw = []

    def __init__(self, start, goal, obstacle_list, rand_area,
                 max_iter=200,
                 connect_circle_dist=50.0
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.connect_circle_dist = connect_circle_dist

        self.curvature = 1.0
        self.goal_yaw_th = np.deg2rad(1.0)
        self.goal_xy_th = 3 #0.5

    def planning(self, animation=True, search_until_max_iter=True):
        """
        planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd)

            if self.check_collision(new_node, self.obstacle_list):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)
                    self.try_goal_path(new_node)

            if animation and i % 5 == 0:
                self.plot_start_goal_arrow()
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    def try_goal_path(self, node):

        goal = self.Node(self.end.x, self.end.y, self.end.yaw)

        new_node = self.steer(node, goal)
        if new_node is None:
            return

        if self.check_collision(new_node, self.obstacle_list):
            self.node_list.append(new_node)

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-1, 36, -1, 36])
        plt.grid(True)
        self.plot_start_goal_arrow()
        plt.pause(0.01)

    def plot_start_goal_arrow(self):
        reeds_shepp_path_planning.plot_arrow(
            self.start.x, self.start.y, self.start.yaw)
        reeds_shepp_path_planning.plot_arrow(
            self.end.x, self.end.y, self.end.yaw)

    def steer(self, from_node, to_node):

        px, py, pyaw, mode, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature)

        if not px:
            return None

        new_node = copy.deepcopy(from_node)
        new_node.x = px[-1]
        new_node.y = py[-1]
        new_node.yaw = pyaw[-1]

        new_node.path_x = px
        new_node.path_y = py
        new_node.path_yaw = pyaw
        new_node.cost += sum([abs(l) for l in course_lengths])
        new_node.parent = from_node

        return new_node

    def calc_new_cost(self, from_node, to_node):

        _, _, _, _, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature)
        if not course_lengths:
            return float("inf")

        return from_node.cost + sum([abs(l) for l in course_lengths])

    def get_random_node(self):

        rnd = self.Node(random.uniform(self.min_rand, self.max_rand),
                        random.uniform(self.min_rand, self.max_rand),
                        random.uniform(-math.pi, math.pi)
                        )

        return rnd

    def search_best_goal_node(self):

        goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.calc_dist_to_goal(node.x, node.y) <= self.goal_xy_th:
                goal_indexes.append(i)
        print("goal_indexes:", len(goal_indexes))

        # angle check
        final_goal_indexes = []
        for i in goal_indexes:
            if abs(self.node_list[i].yaw - self.end.yaw) <= self.goal_yaw_th:
                final_goal_indexes.append(i)

        print("final_goal_indexes:", len(final_goal_indexes))

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        print("min_cost:", min_cost)
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def generate_final_course(self, goal_index):
        path = [[self.end.x, self.end.y, self.end.yaw]]
        node = self.node_list[goal_index]
        while node.parent:
            for (ix, iy, iyaw) in zip(reversed(node.path_x), reversed(node.path_y), reversed(node.path_yaw)):
                path.append([ix, iy, iyaw])
            node = node.parent
        path.append([self.start.x, self.start.y, self.start.yaw])
        return path


def main(max_iter=500):
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [
        (7, 7, 0.5),(7,8,0.5),(7,9,0.5),(7,10,0.5),(8,7,0.5),(8,8,0.5),(8,9,0.5),(8,10,0.5),
        (10, 7, 0.5),(10,8,0.5),(10,9,0.5),(10,10,0.5),(11,7,0.5),(11,8,0.5),(11,9,0.5),(11,10,0.5),
        (13, 7, 0.5),(13,8,0.5),(13,9,0.5),(13,10,0.5),(14,7,0.5),(14,8,0.5),(14,9,0.5),(14,10,0.5),
        (16, 7, 0.5),(16,8,0.5),(16,9,0.5),(16,10,0.5),(17,7,0.5),(17,8,0.5),(17,9,0.5),(17,10,0.5),
        (19, 7, 0.5),(19,8,0.5),(19,9,0.5),(19,10,0.5),(20,7,0.5),(20,8,0.5),(20,9,0.5),(20,10,0.5),
        (22, 7, 0.5),(22,8,0.5),(22,9,0.5),(22,10,0.5),(23,7,0.5),(23,8,0.5),(23,9,0.5),(23,10,0.5),
        (26, 7, 0.5),(26,8,0.5),(26,9,0.5),(26,10,0.5),(25,7,0.5),(25,8,0.5),(25,9,0.5),(25,10,0.5),
        (28, 7, 0.5),(28,8,0.5),(28,9,0.5),(28,10,0.5),(29,7,0.5),(29,8,0.5),(29,9,0.5),(29,10,0.5),

        (7, 18, 0.5),(7,19,0.5),(7,20,0.5),(7,21,0.5),(8,18,0.5),(8,19,0.5),(8,20,0.5),(8,21,0.5),
        (10, 18, 0.5),(10,19,0.5),(10,20,0.5),(10,21,0.5),(11,18,0.5),(11,19,0.5),(11,20,0.5),(11,21,0.5),
        (13, 18, 0.5),(13,19,0.5),(13,20,0.5),(13,21,0.5),(14,18,0.5),(14,19,0.5),(14,20,0.5),(14,21,0.5),
        (16, 18, 0.5),(16,19,0.5),(16,20,0.5),(16,21,0.5),(17,18,0.5),(17,19,0.5),(17,20,0.5),(17,21,0.5),
        (19, 18, 0.5),(19,19,0.5),(19,20,0.5),(19,21,0.5),(20,18,0.5),(20,19,0.5),(20,20,0.5),(20,21,0.5),
        (22, 18, 0.5),(22,19,0.5),(22,20,0.5),(22,21,0.5),(23,18,0.5),(23,19,0.5),(23,20,0.5),(23,21,0.5),
        (26, 18, 0.5),(26,19,0.5),(26,20,0.5),(26,21,0.5),(25,18,0.5),(25,19,0.5),(25,20,0.5),(25,21,0.5),
        (28, 18, 0.5),(28,19,0.5),(28,20,0.5),(28,21,0.5),(29,18,0.5),(29,19,0.5),(29,20,0.5),(29,21,0.5),

        (7, 29, 0.5),(7,30,0.5),(7,31,0.5),(7,32,0.5),(8,29,0.5),(8,30,0.5),(8,31,0.5),(8,32,0.5),
        (10, 29, 0.5),(10,30,0.5),(10,31,0.5),(10,32,0.5),(11,29,0.5),(11,30,0.5),(11,31,0.5),(11,32,0.5),
        (13, 29, 0.5),(13,30,0.5),(13,31,0.5),(13,32,0.5),(14,29,0.5),(14,30,0.5),(14,31,0.5),(14,32,0.5),
        #(16, 29, 0.5),(16,30,0.5),(16,31,0.5),(16,32,0.5),(17,29,0.5),(17,30,0.5),(17,31,0.5),(17,32,0.5),
        (19, 29, 0.5),(19,30,0.5),(19,31,0.5),(19,32,0.5),(20,29,0.5),(20,30,0.5),(20,31,0.5),(20,32,0.5),
        (22, 29, 0.5),(22,30,0.5),(22,31,0.5),(22,32,0.5),(23,29,0.5),(23,30,0.5),(23,31,0.5),(23,32,0.5),
        (26, 29, 0.5),(26,30,0.5),(26,31,0.5),(26,32,0.5),(25,29,0.5),(25,30,0.5),(25,31,0.5),(25,32,0.5),
        (28, 29, 0.5),(28,30,0.5),(28,31,0.5),(28,32,0.5),(29,29,0.5),(29,30,0.5),(29,31,0.5),(29,32,0.5)
    ]  # [x,y,size(radius)]

    for i in range(36):
        obstacleList.append((i,-1,0.5))
        obstacleList.append((i,33,0.5))
    
    for i in range(33):
        obstacleList.append((-1,i,0.5))
        obstacleList.append((36,i,0.5))
    
    for i in range(24):
        obstacleList.append((i+6,11,0.5))
        obstacleList.append((i+6,22,0.5))
        

    # Set Initial parameters
    start = [2.0, 1.5, np.deg2rad(90.0)]
    goal = [16.5, 30.5, np.deg2rad(-90.0)]

    rrt_star_reeds_shepp = RRTStarReedsShepp(start, goal,
                                             obstacleList,
                                             [0, 35], max_iter=max_iter)
    path = rrt_star_reeds_shepp.planning(animation=show_animation)

    # Draw final path
    if path and show_animation:  # pragma: no cover
        rrt_star_reeds_shepp.draw_graph()
        plt.plot([x for (x, y, yaw) in path], [y for (x, y, yaw) in path], '-r')
        plt.grid(True)
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
