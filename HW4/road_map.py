import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from queue import PriorityQueue
from rsplan import path as rspath


# parameter
N_SAMPLE = 250  # number of sample_points
N_KNN = 8  # number of edge from one sampled point
MAX_EDGE_LEN = 50.0  # [m] Maximum edge length

class RoadMap():
    def __init__(self, vehicle, max_point, conversion):
        self.vehicle = vehicle
        self.max_point = max_point  
        self.conversion = conversion
                     
    def build_road_map(self, start, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()   
        else: self.rng = np.random.default_rng(seed=rng)
        
        sample_x, sample_y, sample_heading = [], [], []
        
        while len(sample_x) <= N_SAMPLE:
            tx = (self.rng.random() * self.max_point)
            ty = (self.rng.random() * self.max_point)
            theading = (self.rng.random() * 2*np.pi) - np.pi

            if not self.vehicle.check_collision([tx,ty,theading]) and tx>self.vehicle.collision_r and ty>self.vehicle.collision_r:
                sample_x.append(tx)
                sample_y.append(ty)
                sample_heading.append(theading)
        
        sample_x.append(start[0])
        sample_y.append(start[1])
        sample_heading.append(start[2])
        self.sample_poses = np.vstack((sample_x, sample_y, sample_heading)).T
        
        self.kd_tree = KDTree(self.sample_poses[:,:2])
        
        self.road_map = {}
        self.edge_map = {}

        self.add_edges(self.sample_poses)
        self.local_planner = LocalPlanner(self.road_map, self.edge_map, self.vehicle)
        
        return sample_x, sample_y, sample_heading
    
    def add_edges(self, sample_points):
        edge_i = len(self.edge_map)
        for pose in sample_points:
            node = round_node(pose)
            node_axle = self.vehicle.get_axle_loc(node)
            if node not in self.road_map: self.road_map[node] = {"neighbors": {}}
            _, idxs = self.kd_tree.query(pose[:2], k=2*N_KNN)
            neighbor_poses = self.sample_poses[idxs]
            for neighbor_pose in neighbor_poses:
                neighbor_node = round_node(neighbor_pose)
                if node == neighbor_node: continue
                if neighbor_node not in self.road_map: self.road_map[neighbor_node] = {"neighbors": {}}
                if len(self.road_map[node]["neighbors"]) >= N_KNN or len(self.road_map[neighbor_node]["neighbors"]) >= N_KNN: break
                
                neighbor_axle = self.vehicle.get_axle_loc(neighbor_node)
                plan = rspath(node_axle, neighbor_axle, self.vehicle.turning_r, 0.0, step_size=0.1)
                cost = sum(abs(segment.length) for segment in plan.segments)
                if cost > MAX_EDGE_LEN: break
                
                rs_path = []
                valid = True
                for waypoint in plan.waypoints():
                    if self.vehicle.check_collision([waypoint.x, waypoint.y, waypoint.yaw], axle = True):
                        valid = False
                        break
                    rs_path.append((waypoint.x, waypoint.y, waypoint.yaw))
                    
                if valid:
                    self.edge_map[edge_i] = {'cost': cost, 'path': rs_path}
                    self.road_map[node]["neighbors"][neighbor_node] = edge_i
                    self.road_map[neighbor_node]["neighbors"][node] = edge_i
                    
                    edge_i += 1
    
    def update_road_map_missing(self, obstacle_tree:KDTree, grid_loc, min_dist):
        sample_x, sample_y, sample_heading = [], [], []
        found = False
        while not found:
            tx = (self.rng.random() * self.max_point)
            ty = (self.rng.random() * self.max_point)
            theading = (self.rng.random() * 2*np.pi) - np.pi

            if not self.vehicle.check_collision([tx,ty,theading]) and tx>self.vehicle.collision_r and ty>self.vehicle.collision_r:
                sample_x.append(tx)
                sample_y.append(ty)
                sample_heading.append(theading)
                
            idxs = obstacle_tree.query_ball_point([tx,ty],min_dist)
            for obstacle_point in obstacle_tree.data[idxs]:
                if obstacle_point[0]//self.conversion == grid_loc[1] and obstacle_point[1]//self.conversion == grid_loc[0]:
                    found = True
                    found_point = (tx,ty,theading)
        
        new_samples = np.vstack((sample_x, sample_y, sample_heading)).T
        self.sample_poses = np.vstack((self.sample_poses, new_samples))
        self.kd_tree = KDTree(self.sample_poses[:,:2])
        
        self.add_edges(new_samples)      
        return found_point 
        
    def prm_path(self, start, grid_goal, obstacle_tree:KDTree, radius):
        closest_idxs = self.kd_tree.query_ball_point(((grid_goal[1]+0.5)*self.conversion,(grid_goal[0]+0.5)*self.conversion), 1.5*radius)
        valid_points = []
        for sampled_point in self.sample_poses[closest_idxs]:
            idxs = obstacle_tree.query_ball_point(sampled_point[:2],radius)
            for obstacle_point in obstacle_tree.data[idxs]:
                if obstacle_point[0]//self.conversion == grid_goal[1] and obstacle_point[1]//self.conversion == grid_goal[0]:
                    valid_points.append(sampled_point)
                    
        if len(valid_points) == 0: 
            print("no close points")
            valid_points = [self.update_road_map_missing(obstacle_tree, grid_goal, radius)]
            print(valid_points)
            
        try:
            best_path = []
            for goal in valid_points:
                path = self.local_planner.find_prm_path(start, goal)
                if len(best_path) == 0 or len(best_path) > len(path): best_path = path
            return best_path
        except ValueError:
            print("unable to find path. will update map")
        
        
    
class LocalPlanner():
    def __init__(self, road_map, edge_map, vehicle):
        self.road_map = road_map
        self.edge_map = edge_map
        self.vehicle = vehicle
        
    def find_prm_path(self, start_pose, end_pose):
        frontier = PriorityQueue()
        frontier.put((0,start_pose)) # add start as first node with no cost and came from is None
        came_from = {}
        cost_so_far = {}
        came_from[round_node(start_pose)] = (None, None)
        cost_so_far[round_node(start_pose)] = 0
        end_pose = round_node(end_pose)

        while not frontier.empty():
            item = frontier.get()
            curr_node = round_node(item[1])
            
            if (curr_node) == end_pose: # if it finds the goal, return a formatted path
                return self.format_path(came_from, curr_node)
            
            # Iterates through closets 4 neighbors and see if they are within the field
            for next_node, edge_idx in self.road_map[curr_node]["neighbors"].items():
                edge = self.edge_map[edge_idx]
                new_cost = cost_so_far[curr_node] + edge["cost"]
                prev_cost = cost_so_far.get(next_node)
                if prev_cost is None or new_cost < prev_cost: # only adds to queue if unvisited or cheaper to get to
                    cost_so_far[next_node] = new_cost
                    heuristic = math.sqrt((end_pose[0]-next_node[0])**2 + (end_pose[1]-next_node[1])**2) # Euclidean distance to goal
                    priority = int(new_cost + heuristic)
                    frontier.put((priority,next_node))
                    came_from[next_node] = (curr_node,edge_idx)
        
        raise ValueError("No Path Found")
    
    def format_path(self, came_from, goal_loc):
        path = []
        node, edge_i = came_from[goal_loc]
        while node is not None: # appends nodes in path with goal as beginning
            edge = self.edge_map[edge_i]["path"]
            if round_node(self.vehicle.get_center(edge[0])) != node: edge.reverse()
            path = edge + path
            prev_point = came_from.get(node)
            node, edge_i = prev_point
        return path
    
def round_node(node):
    phi = node[2]
    phi = normalize_angle(phi)
    return (round(node[0],3), round(node[1],3), round(phi, 3))
    
# Normalize angle between pi and -pi
def normalize_angle(phi):
    phi = phi - 2*np.pi if phi > np.pi else phi
    phi = phi + 2*np.pi if phi < -np.pi else phi
    return phi