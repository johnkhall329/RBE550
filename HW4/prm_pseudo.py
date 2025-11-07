import random
from scipy.spatial import KDTree

MAX_EDGES = 1

def check_collision(pose):
    pass

def n_nearest_neighbors(pose):
    pass

def reeds_shepp_path(pose, neighbor):
    pass

class prm():    
    def build_road_map(self, start, N_points):
        sampled_poses = []
        while len(sampled_poses) < N_points:
            pose = random()
            
            if check_collision(pose):
                sampled_poses.append(pose)
            
        sampled_poses.append(start)
        tree = KDTree(sampled_poses)
        
        self.add_edges(sampled_poses)
    
    def add_edges(self, sampled_poses):
        for pose in sampled_poses:
            for neighbor in n_nearest_neighbors(pose):
                if pose.neighbors < MAX_EDGES and neighbor.neighbors < MAX_EDGES:
                    path, cost = reeds_shepp_path(pose, neighbor)
                    if check_collision(path):
                        if path not in self.edge_map:
                            self.edge_map[i] = (cost, path)
        
        for pose in sampled_poses:
            self.connectivity[i] = pose.neighbors/len(self.edge_map)
                
                    