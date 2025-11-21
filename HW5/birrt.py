import enum

import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from tree_geometry import *

class Status(enum.Enum):
    FAILED = 1
    TRAPPED = 2
    ADVANCED = 3
    REACHED = 4

class Tree(object):
    def __init__(self, init, update):
        """
        Tree representation
        :param init: starting location
        :param update: number of vertices added before KDTree updates
        """
        init = convert_to_quat(init)
        self.V = np.array([init])
        self.kd_tree = KDTree(self.V[:,:3])
        self.kd_update = update
        self.V_count = 0
        self.E = {}  # edges in form E[child] = parent

    def update(self, v):
        """
        Add new vertex, will update KDTree if over update threshold
        """
        v = convert_to_quat(v)
        self.V = np.vstack([self.V, v])
        self.V_count += 1
        if self.V_count % self.kd_update == 0:
            self.kd_tree = KDTree(self.V[:,:3])
    
    def nearby(self, x):
        """
        Find closest vertex to random point
        Uses weighted distance to calculate closest to favor orientation
        """
        x = convert_to_quat(x)
        _, idxs = self.kd_tree.query(x[:3], 5)
        min_dist = 1e9
        closest = None
        for idx in np.atleast_1d(idxs):
            if idx >= self.V.shape[0]: continue
            neighbor = self.V[idx]
            dist = weighted_dist(x, neighbor)
            if dist<min_dist:
                min_dist = dist
                closest = neighbor
        return closest
           

class BiRRT():
    def __init__(self, cc, dp, dr, x_init, x_goal, max_samples, res_d, res_r, bound):
        """
        Template RRTConnect planner
        :param cc: collision checking class
        :param dp: position length of edges added to tree
        :param dp: rotation length of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param res_d: resolution of points to sample along edge when checking for collisions
        :param res_r: resolution of points to sample along edge when checking for collisions
        :param bound: max position value that can be sampled
        """
        self.samples_taken = 0
        self.max_samples = max_samples
        self.cc = cc
        self.dp = dp
        self.dr = dr
        self.res_d = res_d
        self.res_r = res_r
        self.x_init = x_init
        self.x_goal = x_goal
        self.bound = bound
        self.trees = []  # list of all trees
        self.add_tree(x_init, 10)  # add initial tree
        self.swapped = False

    def add_tree(self,init, update):
        """
        Create an empty tree and add to trees
        """
        self.trees.append(Tree(init, update))

    def add_vertex(self, tree, v):
        """
        Add vertex to corresponding tree
        :param tree: int, tree to which to add vertex
        :param v: tuple, vertex to add
        """
        # self.trees[tree].V.insert(0, v + v, v)
        self.trees[tree].update(v)
        self.samples_taken += 1  # increment number of samples taken

    def add_edge(self, tree, child, parent):
        """
        Add edge to corresponding tree
        :param tree: int, tree to which to add vertex
        :param child: tuple, child vertex
        :param parent: tuple, parent vertex
        """
        self.trees[tree].E[tuple(np.round(child,3))] = parent


    def get_nearest(self, tree, x):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """
        return self.trees[tree].nearby(x)

    def swap_trees(self):
        """
        Swap trees only
        """
        # swap trees
        self.trees[0], self.trees[1] = self.trees[1], self.trees[0]
        self.swapped = not self.swapped

    def unswap(self):
        """
        Check if trees have been swapped and unswap
        """
        if self.swapped:
            self.swap_trees()
            
    def collision_free_path(self, start, end):
        points = es_points_along_line(start, end, self.res_d, self.res_r)
        coll_free = all(map(self.cc.collision_free, points))
        return coll_free
            
    def connect_to_point(self, tree, x_a, x_b):
        """
        Connect vertex x_a in tree to vertex x_b
        :param tree: int, tree to which to add edge
        :param x_a: tuple, vertex
        :param x_b: tuple, vertex
        :return: bool, True if able to add edge, False if prohibited by an obstacle
        """
        if self.collision_free_path(x_a, x_b):
            self.add_vertex(tree, x_b)
            self.add_edge(tree, x_b, x_a)
            return True
        return False

    def extend(self, tree, x_rand):
        """
        Find closest point in tree and attempt to extend tree towards points
        """
        x_nearest = self.get_nearest(tree, x_rand)
        x_new = steer(x_nearest, x_rand, self.dp, self.dr)
        if self.connect_to_point(tree, x_nearest, x_new):
            if weighted_dist(x_new,convert_to_quat(x_rand)) < 1e-2:
                return x_new, Status.REACHED
            return x_new, Status.ADVANCED
        return x_new, Status.TRAPPED

    def connect(self, tree, x):
        """
        Attempt to direct tree towards newly sampled point in other tree
        Will extend until tree is trapped or reached the other tree
        """
        S = Status.ADVANCED
        while S == Status.ADVANCED:
            x_new, S = self.extend(tree, x)
        return x_new, S
    
    def reconstruct_path(self, tree, x_init, x_goal):
        """
        Reconstruct path from start to goal
        :param tree: int, tree in which to find path
        :param x_init: tuple, starting vertex
        :param x_goal: tuple, ending vertex
        :return: sequence of vertices from start to goal
        """
        path = [convert_to_quat(x_goal)]
        current = tuple(np.round(x_goal,3))
        init_q = convert_to_quat(x_init)
        if np.allclose(init_q, x_goal, atol=1e-3):
            return path
        while not np.allclose(self.trees[tree].E[current], init_q):
            path.append(self.trees[tree].E[current])
            current = tuple(np.round(self.trees[tree].E[current],3))
        path.append(convert_to_quat(x_init))
        path.reverse()
        return path

    def rrt_connect(self):
        """
        RRTConnect
        :return: set of Vertices; Edges in form: vertex: [neighbor_1, neighbor_2, ...]
        """
        self.add_vertex(0, self.x_init)
        self.add_edge(0, convert_to_quat(self.x_init), None)
        self.add_tree(self.x_goal, 20)
        self.add_vertex(1, self.x_goal)
        self.add_edge(1, convert_to_quat(self.x_goal), None)
        
        while self.samples_taken < self.max_samples:
            x_rand = self.cc.sample_free(self.bound)
            x_new, status = self.extend(0, x_rand)
            if status != Status.TRAPPED:
                x_new, connect_status = self.connect(1, x_new)
                if connect_status == Status.REACHED:
                    self.unswap()
                    first_part = self.reconstruct_path(0, self.x_init, self.get_nearest(0, x_new))
                    second_part = self.reconstruct_path(1, self.x_goal, self.get_nearest(1, x_new))
                    second_part.reverse()
                    return first_part + second_part, self.trees[0].E, self.trees[1].E
            self.swap_trees()
            self.samples_taken += 1
        
        print("reached max samples")
        print(self.trees[0].V_count, self.trees[1].V_count)
        self.unswap()
        return None, self.trees[0].E, self.trees[1].E