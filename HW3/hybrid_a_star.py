import numpy as np
from queue import PriorityQueue
import math 
import cv2
import time
import pygame
    
from unconstrained import Unconstrained
from reeds_shepp import reeds_shepp_path_planning as reeds_shepp
from rsplan import path as rspath

D_HEADING = np.pi/8
RESOLUTION = 30
TURNING_RADIUS = RESOLUTION/D_HEADING
TURN_COST = 15
BACKUP_COST = 10
SHOW_ARROWS = True

def dilate_map(map):
    
    dilute_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    # erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    diluted = cv2.dilate(map, dilute_kernel, iterations=2)
    blurred = cv2.GaussianBlur(diluted, (45,45),0)
    return blurred

# a node is a continous (x,y,phi) tuple
# phi is the heading angle in radians 
# x and y are in pixels, with (0,0) as top left corner

def round_node(node):
    phi = node[2]
    phi = phi - 2*np.pi if phi > np.pi else phi
    phi = phi + 2*np.pi if phi < -np.pi else phi
    return (round(node[0]), round(node[1]), round(phi, 2))
    

def discretize(node, resolution=RESOLUTION, turning_a=D_HEADING):
    """
    Sorts a node into a grid based on resolution and turning angle
    resolution: size of each grid square
    turning_a: angle increment for heading angle

    returns as index of cell location and turning increment
    e.g. (12,9,105 degrees) with resolution 10 and turning_a 15 degrees
    returns (1,0,7)
    cell is 1 in x direction (right), 0 in y direction (down),
    and 7 increments of 15 degrees

    """
    x = round(node[0]/resolution)
    y = round(node[1]/resolution)
    phi = node[2]
    phi = phi - 2*np.pi if phi > np.pi else phi
    phi = phi + 2*np.pi if phi < -np.pi else phi
    phi = round(phi/turning_a)
    return (x,y,phi)

def find_neighbors(node, distance=RESOLUTION, turning_a=D_HEADING):
    x,y,phi,dir = node
    
    match dir:
        case 'L':
            cost = [0, TURN_COST, 2*TURN_COST, BACKUP_COST, BACKUP_COST+TURN_COST, BACKUP_COST+2*TURN_COST]
        case 'S':
            cost = [TURN_COST, 0, TURN_COST, BACKUP_COST+TURN_COST, BACKUP_COST, BACKUP_COST+TURN_COST]
        case 'R':
            cost = [2*TURN_COST, TURN_COST, 0, BACKUP_COST+2*TURN_COST, BACKUP_COST+TURN_COST, BACKUP_COST]
        case 'BL':
            cost = [BACKUP_COST, BACKUP_COST+TURN_COST, BACKUP_COST+2*TURN_COST, 0, TURN_COST, 2*TURN_COST]
        case 'B':
            cost = [BACKUP_COST+TURN_COST, BACKUP_COST, BACKUP_COST+TURN_COST, TURN_COST, 0, TURN_COST]
        case 'BR':
            cost = [BACKUP_COST+2*TURN_COST, BACKUP_COST+TURN_COST, BACKUP_COST, 2*TURN_COST, TURN_COST, 0]
    # straight
    dx = distance * math.cos(phi)
    dy = distance * math.sin(phi)
    straight = (x+dx,y+dy,phi,'S')  
    # back
    back = (x-dx,y-dy,phi,'B')

    # consts for turning
    r = distance/turning_a
    d = 2 * r * math.sin(turning_a/2)
    # d = RESOLUTION # for debugging

    # left
    dx = d * math.cos(phi - turning_a/2)
    dy = d * math.sin(phi - turning_a/2)
    left = (x+dx,y+dy,phi-turning_a,'L')
    
    #back_right
    dx = d * math.cos(np.pi + phi - turning_a/2)
    dy = d * math.sin(np.pi + phi - turning_a/2)
    back_right = (x+dx,y+dy,phi-turning_a,'BR')

    # right
    dx = d * math.cos(phi + turning_a/2)
    dy = d * math.sin(phi + turning_a/2)
    right = (x+dx,y+dy,phi+turning_a,'R')
    
    #back_left
    dx = d * math.cos(np.pi + phi + turning_a/2)
    dy = d * math.sin(np.pi + phi + turning_a/2)
    back_left = (x+dx,y+dy,phi+turning_a,'BL')
    
    return ((left, cost[0]), (straight, cost[1]), (right, cost[2]), (back_left, cost[3]), (back, cost[4]), (back_right, cost[5]))

def get_heuristic(curr_state, curr_discritized, goals, two_d_astar_list: list[Unconstrained], step_size=RESOLUTION):
    h_list = []
    for goal in goals:
        new_rspath = rspath(curr_state[:3], goal, TURNING_RADIUS, 0.0, step_size)
        h_list.append(sum(segment.length for segment in new_rspath.segments))
    for two_d_astar in two_d_astar_list:
        h_list.append(two_d_astar.get_unconstrained_path((curr_discritized[1], curr_discritized[0]),step_size))
    return np.max(h_list)

def check_collision(objects, came_from, cost_so_far, node, car):
    path_img = np.zeros_like(objects)
    path = []
    collided=False
    init_node = node
    while came_from[round_node(node)] is not None:
        path.insert(0,node)
        car_loc = cv2.RotatedRect((node[0]+(math.cos(-np.pi/2-node[2])*car.wheelbase/2), node[1]+(math.sin(-np.pi/2-node[2]))*car.wheelbase/2),(car.height,car.width),np.rad2deg(np.pi/2-node[2]))
        pts = car_loc.points().astype(np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(path_img,pts,(255,255,255))
        mask = cv2.bitwise_and(objects,path_img)
        if np.any(mask): 
            collided=True
            for collided_node in path:   
                cost_so_far[discretize(collided_node)] = 1e9
            break
        node = came_from[round_node(node)]
    # print(len(path))
    return collided, path
    
def check_reeds_shepp(objects, plan, car):
    path_img = np.zeros_like(objects)
    rspath = []
    for waypoint in plan.waypoints():
        car_loc = cv2.RotatedRect((waypoint.x+(math.cos(-np.pi/2-waypoint.yaw)*car.wheelbase/2), waypoint.y+(math.sin(-np.pi/2-waypoint.yaw))*car.wheelbase/2),(car.height,car.width),np.rad2deg(np.pi/2-waypoint.yaw))
        pts = car_loc.points().astype(np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(path_img,pts,(255,255,255))
        mask = cv2.bitwise_and(objects,path_img)
        if np.any(mask):
            return False
        if waypoint.curvature == 0.0:
            dir = 'S' if waypoint.driving_direction == 1 else 'B'
        elif waypoint.curvature < 0:
            dir = 'L' if waypoint.driving_direction == 1 else 'BL'
        else:
            dir = 'R' if waypoint.driving_direction == 1 else 'BR'
        rspath.append((waypoint.x, waypoint.y, waypoint.yaw, dir))
    # last = rspath[-1]
    # rspath[-1] = (last[0], last[1], last[2], 'STOP')
    return rspath
    

def hybrid_a_star_path(start_loc, goal_locs, map, car):
    # car_img, diluted_img = get_bin_road(screen)
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    came_from[round_node(start_loc)] = None
    cost_so_far[discretize(start_loc)] = 0
    frontier.put((0,start_loc))
    goal_discretized = [discretize(goal_loc) for goal_loc in goal_locs]
    twodastar_list = [Unconstrained((goal_disc[1],goal_disc[0]),dilate_map(map)) for goal_disc in goal_discretized]
    # color_map = cv2.cvtColor(map,cv2.COLOR_GRAY2BGR)
    # cv2.circle(color_map,(int(goal_loc[0]),int(goal_loc[1])),3, (0,255,0),-1)
    # cv2.circle(color_map,(int(start_loc[0]),int(start_loc[1])),3, (255,0,0),-1)
    itr = 0
    # collision_prob_const = 5
    init_dist = math.sqrt((goal_locs[0][0]-start_loc[0])**2 + (goal_locs[0][1]-start_loc[1])**2)
    while not frontier.empty():
        item = frontier.get()
        curr_node = item[1]
        curr_discritized = discretize(curr_node)
        # cm = color_map.copy()

        if curr_discritized in goal_discretized: #will need to do correct goal checking
            collided, path = check_collision(map, came_from, cost_so_far, curr_node, car)      
            if not collided: 
                return path, []
            else:
                print('collision')
                continue
        
        # curr_dist  = math.sqrt((goal_loc[0]-curr_node[0])**2 + (goal_loc[1]-curr_node[1])**2)
        curr_dist = item[0]-cost_so_far[curr_discritized]
        prob = min(1,curr_dist/init_dist)
        if (prob < np.random.random()) and item[0] != 0:
            collided, path = check_collision(map, came_from, cost_so_far, curr_node, car)
            if collided: continue
            for goal_loc in goal_locs:
                plan = rspath(curr_node[:3], goal_loc, TURNING_RADIUS, 0.0, RESOLUTION)
                rs_path = check_reeds_shepp(map, plan, car)
                if rs_path:
                    print('successful rs')
                    return path, rs_path
        
        # cv2.circle(color_map,(int(curr_node[0]),int(curr_node[1])),3, (0,0,255),-1)
        for next_neighbor in find_neighbors(curr_node):
            next_node, turn_cost = next_neighbor
            if not (0<=curr_node[0]<map.shape[0] and 0<=curr_node[1]<map.shape[1]): continue
            new_cost = cost_so_far[curr_discritized] + turn_cost + RESOLUTION
            next_discritized = discretize(next_node)
            prev_cost = cost_so_far.get(next_discritized)       
            # if came_from.get(round_node(next_node)) is not None and round_node(came_from[round_node(next_node)]) == round_node(curr_node):
            #     continue
            
            if prev_cost is None or new_cost < prev_cost:
                if came_from.get(round_node(curr_node)) is not None and round_node(came_from[round_node(curr_node)]) == round_node(next_node):
                    # print('he')
                    continue
                cost_so_far[next_discritized] = new_cost
                heuristic = get_heuristic(curr_node,curr_discritized, goal_locs,twodastar_list)
                priority = new_cost + heuristic
                frontier.put((priority,next_node))
                came_from[round_node(next_node)] = curr_node
            #     cv2.circle(color_map,(int(next_node[0]),int(next_node[1])),3, (0,0,255),-1)
            # cv2.imshow('Progress', color_map)
            # cv2.waitKey(1)
        itr+=1
    raise ValueError("Unable to find path")