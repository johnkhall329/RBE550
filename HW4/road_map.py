import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from queue import PriorityQueue
from rsplan import path as rspath


# PRM parameters
N_SAMPLE = 200  # number of sample_points
N_KNN = 8  # number of edge from one sampled point
MAX_EDGE_LEN = 75.0  # maximum edge length m
NEW_SAMPLES = 25 # number of new samples to add if path isn't found

class RoadMap():
    def __init__(self, vehicle, max_point, conversion):
        self.vehicle = vehicle
        self.max_point = max_point  
        self.conversion = conversion
        self.path_attempts = 0
        self.max_attempts = 3
        self.step_size = 0.1
                     
    def build_road_map(self, start, rng=None):
        '''
        Builds a probabilistic road map given a vehicle and starting location
        '''
        if rng is None: # Able to seed consistent road map
            self.rng = np.random.default_rng()   
        else: self.rng = np.random.default_rng(seed=rng)
        
        sample_x, sample_y, sample_heading = [], [], []
        
        while len(sample_x) < N_SAMPLE:
            tx = (self.rng.random() * self.max_point) # choose random pose within field
            ty = (self.rng.random() * self.max_point)
            theading = (self.rng.random() * 2*np.pi) - np.pi

            # Check if pose is in collision
            if not self.vehicle.check_collision([tx,ty,theading]) and tx>self.vehicle.collision_r and ty>self.vehicle.collision_r:
                sample_x.append(tx)
                sample_y.append(ty)
                sample_heading.append(theading)
        
        sample_x.append(start[0])
        sample_y.append(start[1])
        sample_heading.append(start[2])
        self.sample_poses = np.vstack((sample_x, sample_y, sample_heading)).T
        
        # Group sampled poses into a KD Tree
        self.kd_tree = KDTree(self.sample_poses[:,:2])
        
        self.road_map = {}
        self.edge_map = {}

        self.add_edges(self.sample_poses)
        
        return sample_x, sample_y, sample_heading
    
    def add_edges(self, sample_points):
        '''
        Connects nodes to K nearest neighbors using Reeds-Shepp paths
        '''
        edge_i = len(self.edge_map)
        for pose in sample_points:
            node = round_node(pose)
            node_axle = self.vehicle.get_axle_loc(node)
            if node not in self.road_map: self.road_map[node] = {"neighbors": {}}
            _, idxs = self.kd_tree.query(pose[:2], k=2*N_KNN) # find nearest neighbors of pose
            neighbor_poses = self.sample_poses[idxs]
            for neighbor_pose in neighbor_poses:
                neighbor_node = round_node(neighbor_pose)
                if node == neighbor_node: continue
                if neighbor_node not in self.road_map: self.road_map[neighbor_node] = {"neighbors": {}}
                if len(self.road_map[node]["neighbors"]) >= N_KNN or len(self.road_map[neighbor_node]["neighbors"]) >= N_KNN: break # don't continue if max connectivity is reached
                
                neighbor_axle = self.vehicle.get_axle_loc(neighbor_node)
                plan = rspath(node_axle, neighbor_axle, self.vehicle.turning_r, 0.0, step_size=self.step_size) # compute Reeds-Shepp between poses
                cost = sum(abs(segment.length) for segment in plan.segments)
                if cost > MAX_EDGE_LEN: break
                
                rs_path = []
                valid = True
                for waypoint in plan.waypoints(): # determine if path is collision free
                    if self.vehicle.check_collision([waypoint.x, waypoint.y, waypoint.yaw], axle = True):
                        valid = False
                        break
                    rs_path.append(self.vehicle.get_center((waypoint.x, waypoint.y, waypoint.yaw)))
                    
                if valid: # add edge to map and connect nodes in road mpa
                    self.edge_map[edge_i] = {'cost': cost, 'path': rs_path}
                    self.road_map[node]["neighbors"][neighbor_node] = edge_i
                    self.road_map[neighbor_node]["neighbors"][node] = edge_i
                    
                    edge_i += 1
        
        total_edges = 0
        self.connectivity = np.zeros(self.sample_poses.shape[0])
        for i,pose in enumerate(self.sample_poses): # determine which poses are connected less, more likely to be expanded nearby
            node = round_node(pose)
            if len(self.road_map[node]["neighbors"]) >= N_KNN: 
                self.connectivity[i] = 0
            else:
                self.connectivity[i] = len(self.road_map[node]["neighbors"])
                total_edges += len(self.road_map[node]["neighbors"])
        self.connectivity = self.connectivity/total_edges
        
    def update_road_map_missing(self, obstacle_tree:KDTree, obstacle_locs, min_dist):
        '''
        Given a desired obstacle and it's locations, sample until there is a pose within a specified distance
        '''
        goal_xs = []
        goal_ys = []
        for grid_loc in obstacle_locs:    
            goal_ys.append(grid_loc[0])
            goal_xs.append(grid_loc[1])
        mean_loc = [np.mean(goal_xs), np.mean(goal_ys)] # find mean location of obstacle
        sample_x, sample_y, sample_heading = [], [], []
        found = False
        while not found:
            tx = self.rng.normal(mean_loc[0]*self.conversion, min_dist) # sample points using normal distribution around obstacle
            ty = self.rng.normal(mean_loc[1]*self.conversion, min_dist)
            theading = (self.rng.random() * 2*np.pi) - np.pi

            # check for collision at sampled pose
            if not self.vehicle.check_collision([tx,ty,theading]) and tx>self.vehicle.collision_r and ty>self.vehicle.collision_r and tx<self.max_point and ty<self.max_point:
                sample_x.append(tx)
                sample_y.append(ty)
                sample_heading.append(theading)
                
            idxs = obstacle_tree.query_ball_point([tx,ty],min_dist)
            for obstacle_point in obstacle_tree.data[idxs]:
                # if sampled pose is close to desired obstacle break
                if int(obstacle_point[0]/self.conversion) in goal_xs and int(obstacle_point[1]/self.conversion) in goal_ys:
                    found = True
                    found_point = (tx,ty,theading)
        
        # add new samples to existing samples and reform KD Tree
        new_samples = np.vstack((sample_x, sample_y, sample_heading)).T
        self.sample_poses = np.vstack((self.sample_poses, new_samples))
        self.kd_tree = KDTree(self.sample_poses[:,:2])
        
        # add new edges as well
        self.add_edges(new_samples)      
        return found_point
    
    def update_road_map_connectivity(self, n_samples, radius=MAX_EDGE_LEN/4):
        '''
        If a path isn't possible to a pose, take more samples around less connected nodes
        Only takes a certain number of samples
        '''
        sample_x, sample_y, sample_heading = [], [], []
        sample_idxs = self.rng.choice(self.sample_poses.shape[0], size=n_samples, replace=False, p=self.connectivity)
        for sample_i in sample_idxs:
            prev_x, prev_y, _ = self.sample_poses[sample_i] 
            change=self.rng.random() * 2*np.pi
            new_x = prev_x + radius*math.cos(change) # create a point somewhere around previously sampled point with a radius
            new_y = prev_y + radius*math.sin(change)
            found = False
            while not found:
                tx = self.rng.normal(new_x, MAX_EDGE_LEN/3) # sample using a normal distribution around new point
                ty = self.rng.normal(new_y, MAX_EDGE_LEN/3)
                theading = (self.rng.random() * 2*np.pi) - np.pi

                # check for collision of point
                if not self.vehicle.check_collision([tx,ty,theading]) and tx>self.vehicle.collision_r and ty>self.vehicle.collision_r and tx<self.max_point and ty<self.max_point:
                    sample_x.append(tx)
                    sample_y.append(ty)
                    sample_heading.append(theading)
                    found = True
                    
        # add new samples and rebuild KD Tree
        new_samples = np.vstack((sample_x, sample_y, sample_heading)).T
        self.sample_poses = np.vstack((self.sample_poses, new_samples))
        self.kd_tree = KDTree(self.sample_poses[:,:2])
        
        self.add_edges(new_samples)
            
        
    def prm_path(self, start, obstacle_groups, obstacle_tree:KDTree, radius, updated=False):
        '''
        Determines poses close to desired obstacles and then finds the shortest path
        '''
        valid_points = {}
        for obstacle_locs in obstacle_groups:
            for grid_goal in obstacle_locs: # find nearby poses and determine if they are within the radius to extinguish
                closest_idxs = self.kd_tree.query_ball_point(((grid_goal[1]+0.5)*self.conversion,(grid_goal[0]+0.5)*self.conversion), 1.5*radius)
                for sampled_point in self.sample_poses[closest_idxs]:
                    idxs = obstacle_tree.query_ball_point(sampled_point[:2],radius)
                    for obstacle_point in obstacle_tree.data[idxs]:
                        if obstacle_point[0]//self.conversion == grid_goal[1] and obstacle_point[1]//self.conversion == grid_goal[0]:
                            if round_node(sampled_point) not in valid_points:
                                valid_points[round_node(sampled_point)] = sampled_point
                    
        if len(valid_points) == 0: # if there are no points, update road map to find locations nearby
            print("no close points")
            for group in obstacle_groups:
                new_point = self.update_road_map_missing(obstacle_tree, group, radius)
                valid_points[round_node(new_point)] = new_point
            updated = True
            
        best_path = []
        best_path_len = 0
        found_path = False
        self.local_planner = LocalPlanner(self.road_map, self.edge_map, self.vehicle, start) # create local planner object based on current location
        for goal in valid_points.values():
            try:
                path, length = self.local_planner.find_prm_path(goal) # use local planner to find path to shortest path to an obstacle
                if best_path_len == 0 or length < best_path_len: 
                    best_path = path
                    best_path_len = length
                    found_path = True
                    self.vehicle.goal = grid_goal
            except ValueError:
                pass
        if found_path:        
            self.path_attempts = 0
            return best_path, updated
        else:
            # If no path is found, update map and reattempt until path is found or max attempts is reached. Will do nothing if max is reached
            if self.path_attempts >= self.max_attempts:
                print("Unable to connect path")
                self.path_attempts = 0
                return None, False
            self.path_attempts += 1
            print("Updating map to hopefully improve connectivity")
            self.update_road_map_connectivity(NEW_SAMPLES) # add N new samples to map
            return self.prm_path(start, obstacle_groups, obstacle_tree, radius, True)
        
    
class LocalPlanner():
    def __init__(self, road_map, edge_map, vehicle, start_pose):
        self.road_map = road_map
        self.edge_map = edge_map
        self.vehicle = vehicle
        self.start_pose = start_pose
        self.frontier = PriorityQueue()
        self.frontier.put((0,self.start_pose))
        self.came_from = {}
        self.cost_so_far = {}
        self.came_from[round_node(self.start_pose)] = (None, None)
        self.cost_so_far[round_node(self.start_pose)] = 0
        
    def find_prm_path(self, end_pose):
        '''
        Using Dijkstra's, find path to desired pose
        '''
        end_pose = round_node(end_pose)
        if end_pose in self.cost_so_far:
            return self.format_path(end_pose)

        while not self.frontier.empty():
            item = self.frontier.get()
            curr_node = round_node(item[1])
            
            if curr_node == end_pose: # if it finds the goal, return a formatted path
                return self.format_path(curr_node)
            
            # Iterates through connected nodes
            for next_node, edge_idx in self.road_map[curr_node]["neighbors"].items():
                edge = self.edge_map[edge_idx]
                new_cost = self.cost_so_far[curr_node] + edge["cost"]
                prev_cost = self.cost_so_far.get(next_node)
                if prev_cost is None or new_cost < prev_cost: # only adds to queue if unvisited or cheaper to get to
                    self.cost_so_far[next_node] = new_cost
                    priority = new_cost
                    self.frontier.put((priority,next_node))
                    self.came_from[next_node] = (curr_node,edge_idx)
        
        raise ValueError("No Path Found")
    
    def format_path(self, goal_loc):
        path = []
        length = 0
        node, edge_i = self.came_from[goal_loc]
        while node is not None: # appends nodes in path with goal as beginning
            edge = self.edge_map[edge_i]["path"]
            if round_node(edge[0]) != node: edge.reverse() # some rs-paths are reversed
            path.insert(0,edge[1:])
            length += self.edge_map[edge_i]["cost"]
            prev_point = self.came_from.get(node)
            node, edge_i = prev_point
        return path, length
    
# Round to eliminate floating point error
def round_node(node):
    phi = node[2]
    phi = normalize_angle(phi)
    return (round(node[0],3), round(node[1],3), round(phi, 3))
    
# Normalize angle between pi and -pi
def normalize_angle(phi):
    phi = phi - 2*np.pi if phi > np.pi else phi
    phi = phi + 2*np.pi if phi < -np.pi else phi
    return phi