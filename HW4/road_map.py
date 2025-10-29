import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from rsplan import path as rspath


# parameter
N_SAMPLE = 250  # number of sample_points
N_KNN = 5  # number of edge from one sampled point
MAX_EDGE_LEN = 50.0  # [m] Maximum edge length

class RoadMap():
    def __init__(self, vehicle, max_point, conversion):
        self.vehicle = vehicle
        self.max_point = max_point  
        self.conversion = conversion
    
    def update_road_map_missing(self, obstacle_tree:KDTree, grid_loc, min_dist):
        sample_x, sample_y, sample_heading = [], [], []
        found = False
        while not found:
            tx = (np.random.random() * self.max_point)
            ty = (np.random.random() * self.max_point)
            theading = (np.random.random() * 2*np.pi) - np.pi

            if not self.vehicle.check_collision([tx,ty,theading]) and tx>self.vehicle.collision_r and ty>self.vehicle.collision_r:
                sample_x.append(tx)
                sample_y.append(ty)
                sample_heading.append(theading)
                
            idxs = obstacle_tree.query_ball_point([tx,ty],min_dist)
            for obstacle_point in obstacle_tree.data[idxs]:
                if obstacle_point[0]//self.conversion == grid_loc[0] and obstacle_point[1]//self.conversion == grid_loc[1]:
                    found = True
        
        new_samples = np.vstack((sample_x, sample_y, sample_heading)).T
        self.sample_poses = np.vstack((self.sample_poses, new_samples))
        self.kd_tree = KDTree(self.sample_poses[:,:2])
        
        edge_i = len(self.edge_map)
        for pose in new_samples:
            node = self.round_node(pose)
            if node not in self.road_map: self.road_map[node] = {"neighbors": {}}
            _, idxs = self.kd_tree.query(pose[:2], k=2*N_KNN)
            neighbor_poses = self.sample_poses[idxs]
            for neighbor_pose in neighbor_poses:
                neighbor_node = self.round_node(neighbor_pose)
                if node == neighbor_node: continue
                if neighbor_node not in self.road_map: self.road_map[neighbor_node] = {"neighbors": {}}
                if len(self.road_map[node]["neighbors"]) >= N_KNN or len(self.road_map[neighbor_node]["neighbors"]) >= N_KNN: break
                
                plan = rspath(node, neighbor_node, self.vehicle.turning_r, 0.0, step_size=0.1)
                cost = sum(segment.length for segment in plan.segments)
                if cost > MAX_EDGE_LEN: break
                
                rs_path = []
                valid = True
                for waypoint in plan.waypoints():
                    if self.vehicle.check_collision([waypoint.x, waypoint.y, waypoint.yaw]):
                        valid = False
                        break
                    rs_path.append((waypoint.x, waypoint.y, waypoint.yaw))
                    
                if valid:
                    self.edge_map[edge_i] = {'cost': cost, 'path': rs_path}
                    self.road_map[node]["neighbors"][neighbor_node] = edge_i
                    self.road_map[neighbor_node]["neighbors"][node] = edge_i
                    edge_i += 1         
                    
    # def update_road_map_connection(self, sample_point):
    #     sample_node = self.round_node(sample_point)
                     
    
    def build_road_map(self, start):
        sample_x, sample_y, sample_heading = [], [], []
        
        while len(sample_x) <= N_SAMPLE:
            tx = (np.random.random() * self.max_point)
            ty = (np.random.random() * self.max_point)
            theading = (np.random.random() * 2*np.pi) - np.pi

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
        edge_i = 0
        for pose in self.sample_poses:
            node = self.round_node(pose)
            if node not in self.road_map: self.road_map[node] = {"neighbors": {}}
            _, idxs = self.kd_tree.query(pose[:2], k=2*N_KNN)
            neighbor_poses = self.sample_poses[idxs]
            for neighbor_pose in neighbor_poses:
                neighbor_node = self.round_node(neighbor_pose)
                if node == neighbor_node: continue
                if neighbor_node not in self.road_map: self.road_map[neighbor_node] = {"neighbors": {}}
                if len(self.road_map[node]["neighbors"]) >= N_KNN or len(self.road_map[neighbor_node]["neighbors"]) >= N_KNN: break
                
                plan = rspath(node, neighbor_node, self.vehicle.turning_r, 0.0, step_size=0.1)
                cost = sum(segment.length for segment in plan.segments)
                if cost > MAX_EDGE_LEN: break
                
                rs_path = []
                valid = True
                for waypoint in plan.waypoints():
                    if self.vehicle.check_collision([waypoint.x, waypoint.y, waypoint.yaw]):
                        valid = False
                        break
                    rs_path.append((waypoint.x, waypoint.y, waypoint.yaw))
                    
                if valid:
                    self.edge_map[edge_i] = {'cost': cost, 'path': rs_path}
                    self.road_map[node]["neighbors"][neighbor_node] = edge_i
                    self.road_map[neighbor_node]["neighbors"][node] = edge_i
                    edge_i += 1                           
                    
        print(edge_i)
        return sample_x, sample_y, sample_heading
    
    def round_node(self, node):
        phi = node[2]
        phi = normalize_angle(phi)
        return (round(node[0],3), round(node[1],3), round(phi, 3))
    
# Normalize angle between pi and -pi
def normalize_angle(phi):
    phi = phi - 2*np.pi if phi > np.pi else phi
    phi = phi + 2*np.pi if phi < -np.pi else phi
    return phi