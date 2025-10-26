import numpy as np
from queue import PriorityQueue
import math 
import cv2
import time
    
from unconstrained import Unconstrained
from rsplan import path as rspath

TURN_COST = 10
BACKUP_COST = 25
JACKKNIFE_COST = 20
SHOW_ARROWS = True

class HybridAStar():
    def __init__(self, map, car, resolution, ang_resolution):
        self.map = map
        self.car = car
        self.resolution = resolution
        self.ang_resolution = ang_resolution
        self.turning_r = resolution/ang_resolution #if car.type != 'delivery' else 1
        
    def hybrid_a_star_path(self, start_loc, goal_locs):
        self.frontier = PriorityQueue()
        self.came_from = {}
        self.cost_so_far = {}
        self.came_from[self.round_node(start_loc)] = None
        self.cost_so_far[self.discretize(start_loc)] = 0
        self.frontier.put((0,start_loc))
        self.goal_locs = goal_locs
        self.start_loc = start_loc
        goal_discretized = [self.discretize(goal_loc) for goal_loc in self.goal_locs]
        self.twod_astar_list = [Unconstrained((goal_disc[1],goal_disc[0]),dilate_map(self.map)) for goal_disc in goal_discretized] # create Unconstrained A* planners for each goal location
        color_map = cv2.cvtColor(self.map,cv2.COLOR_GRAY2BGR)
        # cv2.circle(color_map,(int(goal_locs[0][0]),int(goal_locs[0][1])),3, (0,255,0),-1)
        # cv2.circle(color_map,(int(start_loc[0]),int(start_loc[1])),3, (255,0,0),-1)
        itr = 0
        init_dist = math.sqrt((goal_locs[0][0]-start_loc[0])**2 + (goal_locs[0][1]-start_loc[1])**2)
        while not self.frontier.empty():
            item = self.frontier.get()
            curr_node = item[1]
            curr_discretized = self.discretize(curr_node)
            # cm = color_map.copy()

            if curr_discretized in goal_discretized: # goal checking
                collided, path = self.check_collision(curr_node, False) # determine if path collides at any point
                if not collided: 
                    # Get correct goal and use Reeds Shepp path to get to the exact location
                    if self.car.type != 'truck':
                        goal_idx = goal_discretized.index(curr_discretized)
                        rs_goal = goal_locs[goal_idx]
                        plan = rspath(curr_node[:3], rs_goal, self.turning_r, 0.0, self.resolution)
                        rs_path = self.check_reeds_shepp(plan, curr_node)
                    else:
                        rs_path = []
                    return path, rs_path
                else:
                    print('collision')
                    continue
            
            collided, _ = self.check_collision(curr_node) # check if path to current node has collisions
            if collided: 
                # print('collided')
                continue
            
            # Use the ratio of the distance left and initial distance as a probability for collsion checking and reeds shepp analytic expansion
            curr_dist = item[0]-self.cost_so_far[curr_discretized]
            prob = curr_dist/init_dist
            if (prob < np.random.random()) and item[0] != 0 and self.car.type != 'truck':
                plans = {}
                for goal_loc in goal_locs:
                    plan = rspath(curr_node[:3], goal_loc, self.turning_r, 0.0, self.resolution)
                    plan_cost = sum(segment.length for segment in plan.segments)
                    plans[plan_cost] = plan
                for plan_id in sorted(plans.keys()):
                    rs_path = self.check_reeds_shepp(plans[plan_id], curr_node)
                if len(rs_path)!=0:
                    print('successful rs')
                    _, path = self.check_collision(curr_node, False)
                    return path, rs_path
                # pass
            
            # cv2.circle(color_map,(int(curr_node[0]),int(curr_node[1])),3, (0,0,255),-1)
            for next_neighbor in self.find_neighbors(curr_node):
                next_node, turn_cost = next_neighbor
                if not (0<=curr_node[0]<self.map.shape[0] and 0<=curr_node[1]<self.map.shape[1]): continue # skip nodes generated outside of the map
                new_cost = self.cost_so_far[curr_discretized] + turn_cost + self.resolution
                next_discritized = self.discretize(next_node)
                prev_cost = self.cost_so_far.get(next_discritized)       
                
                if prev_cost is None or new_cost < prev_cost:
                    if self.came_from.get(self.round_node(curr_node)) is not None and self.round_node(self.came_from[self.round_node(curr_node)]) == self.round_node(next_node):
                        # print('he')
                        continue
                    self.cost_so_far[next_discritized] = new_cost
                    heuristic = self.get_heuristic(curr_node,curr_discretized)
                    priority = new_cost + heuristic
                    self.frontier.put((priority,next_node))
                    self.came_from[self.round_node(next_node)] = curr_node
                    # cv2.circle(color_map,(int(next_node[0]),int(next_node[1])),3, (0,0,255),-1)
                # cv2.imshow('Progress', color_map)
                # cv2.waitKey(1)
            itr+=1
        raise ValueError("Unable to find path")

    def discretize(self, node):
        """
        Sorts a node into a grid based on resolution and turning angle
        """
        x = round(node[0]/self.resolution)
        y = round(node[1]/self.resolution)
        phi = node[2]
        phi = normalize_angle(phi)
        phi = round(phi/self.ang_resolution)
        if self.car.type == 'truck':
            tphi = node[3]
            tphi = normalize_angle(tphi)
            tphi = round(tphi/self.ang_resolution)
            return (x,y,phi,tphi)
        else:
            return (x,y,phi)

    def find_neighbors(self, node):
        """
        Finds the six neighbors of a node and calculates the additional cost to arrive at that node
        """
        if self.car.type != 'truck':
            x,y,phi,dir = node
        else:
            x,y,phi,tphi,dir = node
        
        dist_to_goal = 1e9
        for goal in self.goal_locs:
            dist_to_goal = min(dist_to_goal, np.hypot(node[0]-goal[0], node[1]-goal[1]))
        
        if dist_to_goal < 100:
            back_up_cost = BACKUP_COST/2
        else:
            back_up_cost = BACKUP_COST
        # Punishes changing direction or turning angle. Double for changing from left-right and vice-versa
        match dir:
            case 'L':
                cost = [0, TURN_COST, TURN_COST, back_up_cost, back_up_cost+TURN_COST, back_up_cost+TURN_COST]
            case 'S':
                cost = [TURN_COST, 0, TURN_COST, back_up_cost+TURN_COST, back_up_cost, back_up_cost+TURN_COST]
            case 'R':
                cost = [TURN_COST, TURN_COST, 0, back_up_cost+TURN_COST, back_up_cost+TURN_COST, back_up_cost]
            case 'BL':
                cost = [back_up_cost, back_up_cost+TURN_COST, back_up_cost+TURN_COST, 0, TURN_COST, TURN_COST]
            case 'B':
                cost = [back_up_cost+TURN_COST, back_up_cost, back_up_cost+TURN_COST, TURN_COST, 0, TURN_COST]
            case 'BR':
                cost = [back_up_cost+TURN_COST, back_up_cost+TURN_COST, back_up_cost, TURN_COST, TURN_COST, 0]
        # straight
        dx = self.resolution * math.cos(phi)
        dy = self.resolution * math.sin(phi)
        if self.car.type == 'truck':
            dtphi = self.resolution/self.car.trailer_dist*math.sin(tphi-phi) # Kinematics for trailer heading
            new_tphi = normalize_angle(tphi-dtphi)
            cost[1] += JACKKNIFE_COST*abs(phi-new_tphi)
            straight = (x+dx,y+dy,phi,new_tphi, 'S')
        else:
            straight = (x+dx,y+dy,phi,'S')
          
        # back
        if self.car.type == 'truck':
            dtphi = -self.resolution/self.car.trailer_dist*math.sin(tphi-phi)
            new_tphi = normalize_angle(tphi-dtphi)
            cost[4] += JACKKNIFE_COST*abs(phi-new_tphi)
            back = (x-dx,y-dy,phi,new_tphi, 'B')
        else:
            back = (x-dx,y-dy,phi,'B')

        # consts for turning
        r = self.resolution/self.ang_resolution
        d = 2 * r * math.sin(self.ang_resolution/2)

        # left
        dx = d * math.cos(phi - self.ang_resolution/2)
        dy = d * math.sin(phi - self.ang_resolution/2)
        
        new_phi = normalize_angle(phi-self.ang_resolution)
        if self.car.type == 'truck':
            dtphi = self.resolution/self.car.trailer_dist*math.sin(tphi-(new_phi+phi)/2)
            new_tphi = normalize_angle(tphi-dtphi)
            cost[0] += JACKKNIFE_COST*abs(new_phi-new_tphi)
            left = (x+dx,y+dy,new_phi,new_tphi,'L')
        else:
            left = (x+dx,y+dy,new_phi,'L')
        
        #back_right
        dx = d * math.cos(np.pi + phi - self.ang_resolution/2)
        dy = d * math.sin(np.pi + phi - self.ang_resolution/2)
        
        new_phi = normalize_angle(phi-self.ang_resolution)
        if self.car.type == 'truck':
            dtphi = -self.resolution/self.car.trailer_dist*math.sin(tphi-(new_phi+phi)/2)
            new_tphi = normalize_angle(tphi-dtphi)
            cost[5] += JACKKNIFE_COST*abs(new_phi-new_tphi)
            back_right = (x+dx,y+dy,new_phi,new_tphi,'BR')
        else:
            back_right = (x+dx,y+dy,new_phi,'BR')
        # right
        dx = d * math.cos(phi + self.ang_resolution/2)
        dy = d * math.sin(phi + self.ang_resolution/2)

        new_phi = normalize_angle(phi+self.ang_resolution)
        if self.car.type == 'truck':
            dtphi = self.resolution/self.car.trailer_dist*math.sin(tphi-(new_phi+phi)/2)
            new_tphi = normalize_angle(tphi-dtphi)
            cost[2] += JACKKNIFE_COST*abs(new_phi-new_tphi)
            right = (x+dx,y+dy,new_phi,new_tphi,'R')
        else:
            right = (x+dx,y+dy,new_phi,'R')
        
        #back_left
        dx = d * math.cos(np.pi + phi + self.ang_resolution/2)
        dy = d * math.sin(np.pi + phi + self.ang_resolution/2)
        
        new_phi = normalize_angle(phi+self.ang_resolution)
        if self.car.type == 'truck':
            dtphi = -self.resolution/self.car.trailer_dist*math.sin(tphi-(new_phi+phi)/2)
            new_tphi = normalize_angle(tphi-dtphi)
            cost[3] += JACKKNIFE_COST*abs(new_phi-new_tphi)
            back_left = (x+dx,y+dy,new_phi,new_tphi,'BL')
        else:
            back_left = (x+dx,y+dy,new_phi,'BL')
        
        return [(left, cost[0]), (straight, cost[1]), (right, cost[2]), (back_left, cost[3]), (back, cost[4]), (back_right, cost[5])]

    def get_heuristic(self, curr_state, curr_discretized):
        h_list = []
        for goal_idx, goal in enumerate(self.goal_locs): # For each goal location, get reeds-shepp path length and unconstrained cost
            new_rspath = rspath(curr_state[:3], goal, self.turning_r, 0.0, self.resolution)
            astar_h = self.twod_astar_list[goal_idx].get_unconstrained_path((curr_discretized[1], curr_discretized[0]),self.resolution)
            h_list.append(max(sum(segment.length for segment in new_rspath.segments), astar_h)) # Take max of the two heuristics
        return np.max(h_list)

    def check_collision(self, node, individual = True):
        """
        From current node, trace path and determine if there are any collisions present.
        This includes self collisions in the case of a truck.
        
        Will set nodes in path after a collision to a high cost under the assumption they can be reach
        """
        path_img = np.zeros_like(self.map)
        collided=False
        
        if individual:
            path = [node]
        else:
            path = []
            # Reorder path from start to current node
            while self.came_from[self.round_node(node)] is not None:
                path.insert(0,node)
                node = self.came_from[self.round_node(node)]
                if len(path) > len(self.came_from): # Prevent circular time looping. Not handled very well at the moment
                    print('circular?')
                    return True, []
        
        if self.car.type == 'truck': tphi = self.start_loc[3]
        
        # Check each node in path to see if there is a collision
        for i, node in enumerate(path):
            # Draw rectangle of vehicle
            car_loc = cv2.RotatedRect((node[0]+(math.cos(node[2])*self.car.wheelbase/2), node[1]+(math.sin(node[2]))*self.car.wheelbase/2),(self.car.height,self.car.width),np.rad2deg(node[2]))
            pts = car_loc.points().astype(np.int32).reshape((-1, 1, 2))
            if self.car.type == 'truck':
                # If it is a truck, check for self collisions first
                self_collision_truck = np.zeros_like(self.map)
                self_collision_trailer = np.zeros_like(self.map)
                dtphi = self.resolution/self.car.trailer_dist*math.sin(tphi-node[3]) if 'B' not in node[4] else -self.resolution/self.car.trailer_dist*math.sin(tphi-node[3])
                tphi = normalize_angle(tphi-dtphi)
                trailer_center = (node[0]-(self.car.trailer_dist*math.cos(tphi)), node[1]-(self.car.trailer_dist*math.sin(tphi)))
                trailer_rect = cv2.RotatedRect(trailer_center, (self.car.trailer_height, self.car.trailer_width), np.rad2deg(tphi))
                t_pts = trailer_rect.points().astype(np.int32).reshape((-1, 1, 2))
                cv2.fillConvexPoly(self_collision_truck,pts,(255,255,255))
                cv2.fillConvexPoly(self_collision_trailer,t_pts,(255,255,255))
                if np.any(cv2.bitwise_and(self_collision_trailer, self_collision_truck)):
                    collided = True
                path_img = cv2.bitwise_or(path_img, cv2.bitwise_or(self_collision_trailer, self_collision_truck))
            else:
                cv2.fillConvexPoly(path_img,pts,(255,255,255))
            mask = cv2.bitwise_and(self.map,path_img)
            if np.any(mask): 
                collided=True
                # If there is a collsision, go through nodes after collision and assign high cost in case there is a safe way to reach them
                for collided_node in path[i:]:   
                    self.cost_so_far[self.discretize(collided_node)] = 1e9
                break
            node = self.came_from[self.round_node(node)]
        # print(len(path))
        return collided, path
        
    def check_reeds_shepp(self, plan, curr_node):
        """
        Almost exactly the same as check collision, except for with a path generated by reeds-shepp
        Will also create the direction between each node for interpolation
        """
        path_img = np.zeros_like(self.map)
        # path_img = objects.copy()
        rspath = []
        tphi = curr_node[3]
        for waypoint in plan.waypoints():
            car_loc = cv2.RotatedRect((waypoint.x+(math.cos(waypoint.yaw)*self.car.wheelbase/2), waypoint.y+(math.sin(waypoint.yaw))*self.car.wheelbase/2),(self.car.height,self.car.width),np.rad2deg(waypoint.yaw))
            pts = car_loc.points().astype(np.int32).reshape((-1, 1, 2))
            if self.car.type == 'truck':
                self_collision_truck = np.zeros_like(self.map)
                self_collision_trailer = np.zeros_like(self.map)
                dtphi = self.resolution/self.car.trailer_dist*math.sin(tphi-waypoint.yaw) if waypoint.driving_direction == 1 else -self.resolution/self.car.trailer_dist*math.sin(tphi-waypoint.yaw)
                tphi = normalize_angle(tphi-dtphi)
                trailer_center = (waypoint.x-(self.car.trailer_dist*math.cos(tphi)), waypoint.y-(self.car.trailer_dist*math.sin(tphi)))
                trailer_rect = cv2.RotatedRect(trailer_center, (self.car.trailer_height, self.car.trailer_width), np.rad2deg(tphi))
                t_pts = trailer_rect.points().astype(np.int32).reshape((-1, 1, 2))
                cv2.fillConvexPoly(self_collision_truck,pts,(255,255,255))
                cv2.fillConvexPoly(self_collision_trailer,t_pts,(255,255,255))
                if np.any(cv2.bitwise_and(self_collision_trailer, self_collision_truck)):
                    return []
                path_img = cv2.bitwise_or(path_img, cv2.bitwise_or(self_collision_trailer, self_collision_truck))
            else:
                cv2.fillConvexPoly(path_img,pts,(255,255,255))

            mask = cv2.bitwise_and(self.map,path_img)
            if np.any(mask):
                return []
            if waypoint.curvature == 0.0:
                dir = 'S' if waypoint.driving_direction == 1 else 'B'
            elif waypoint.curvature < 0:
                dir = 'L' if waypoint.driving_direction == 1 else 'BL'
            else:
                dir = 'R' if waypoint.driving_direction == 1 else 'BR'
            if self.car.type == 'truck':
                rspath.append((waypoint.x, waypoint.y, waypoint.yaw, tphi, dir))
            else:
                rspath.append((waypoint.x, waypoint.y, waypoint.yaw, dir))
        # last = rspath[-1]
        # rspath[-1] = (last[0], last[1], last[2], 'STOP')
        return rspath


    # a node is a continous (x,y,phi) tuple
    # phi is the heading angle in radians 
    # x and y are in pixels, with (0,0) as top left corner
    def round_node(self, node):
        phi = node[2]
        phi = normalize_angle(phi)
        if self.car.type == 'truck':
            tphi = node[3]
            tphi = normalize_angle(tphi)
            return (round(node[0]), round(node[1]), round(phi, 2), round(tphi, 2))
        else:
            return (round(node[0]), round(node[1]), round(phi, 2))
        
# create dilated map for 2D A*        
def dilate_map(map):
    dilute_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    diluted = cv2.dilate(map, dilute_kernel, iterations=2)
    blurred = cv2.GaussianBlur(diluted, (35,35),0)
    return blurred

# Normalize angle between pi and -pi
def normalize_angle(phi):
    phi = phi - 2*np.pi if phi > np.pi else phi
    phi = phi + 2*np.pi if phi < -np.pi else phi
    return phi