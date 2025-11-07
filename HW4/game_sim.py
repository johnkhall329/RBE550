import cv2
import numpy as np
import time
import math
from scipy.spatial import KDTree

from obstacle_field import FieldCreator
from firetruck import Firetruck
from road_map import RoadMap
from wumpus import Wumpus, find_neighbors


MAP = 50
GRID_SIZE_M = 5
GRID_SIZE_PX = 15
PXS_PER_M = GRID_SIZE_PX//GRID_SIZE_M

EXTINGUISHING = 'ex'
DRIVING = 'dr'
IDLE = 'id'

def display_map(field, map, edges = False):
    '''
    Taking a PRM, display the poses and connecting edges if desired
    '''
    for point in map.sample_poses:
        sample_box = cv2.RotatedRect((point[0],point[1]),(firetruck.length,firetruck.width),np.rad2deg(point[2]))
        pts = sample_box.points()*PXS_PER_M
        axle_loc = firetruck.get_axle_loc(point)
        cv2.drawContours(field,[pts.astype(np.int32).reshape((-1, 1, 2))],0,(255,0,0),1)
        cv2.arrowedLine(field,(round(axle_loc[0]*PXS_PER_M),round(axle_loc[1]*PXS_PER_M)),
                        (round(point[0]*PXS_PER_M), round(point[1]*PXS_PER_M)),(0,0,255),2)
    if edges:    
        for edge in map.edge_map.values():
            path = edge["path"]
            for path_point in path:
                cv2.circle(field, (round(path_point[0]*PXS_PER_M), round(path_point[1]*PXS_PER_M)),1, (0,0,0),1)

def display_path(field, path, skip):
    '''
    Display the current path of the Wumpus or Firetruck
    skip determines how frequently to display a point, the firetruck is every 0.1m which is too frequent
    '''
    if len(path) == 0: return
    elif isinstance(path[0],list):
        for segment in path:
            display_path(field, segment, skip)
    else:
        for i in range(0, len(path), skip):
            point = path[i]
            cv2.circle(field, (round(point[0]*PXS_PER_M), round(point[1]*PXS_PER_M)), 1, (0,0,0), -1)
        cv2.circle(field, (round(path[0][0]*PXS_PER_M), round(path[0][1]*PXS_PER_M)), 5, (255,0,0), 1)
        cv2.circle(field, (round(path[-1][0]*PXS_PER_M), round(path[-1][1]*PXS_PER_M)), 5, (0,255,0), 1)

if __name__ == '__main__':
    field = FieldCreator()
    field_time = time.time()
    field.createField(0.1, MAP, GRID_SIZE_PX)
    print(len(field.obstacle_groups))
    
    # Convert field image into a kdtree with obstacle locations in m
    obstacles_pxs = np.where(cv2.cvtColor(field.field, cv2.COLOR_BGR2GRAY)<255)
    obstacles_m = np.vstack([obstacles_pxs[1], obstacles_pxs[0]]).T/PXS_PER_M
    kdtree = KDTree(obstacles_m)
    print('field time: ', time.time() - field_time)

    # Firetruck and Wumpus starting locations
    firetruck_start = (127.5,247.5,np.deg2rad(-90))
    firetruck = Firetruck(firetruck_start, kdtree, PXS_PER_M)
    
    wumpus_start = (127.5,2.5)
    wumpus = Wumpus(wumpus_start, field.small_field, GRID_SIZE_M)

    road_time = time.time()
    road_map = RoadMap(firetruck, (MAP-1)*GRID_SIZE_M, GRID_SIZE_M)
    road_map.build_road_map(firetruck_start)
    road_time_final=time.time() - road_time
    print('Road Map Creation Time: ', road_time_final)
    firetruck.time += road_time_final
    
    display_map(field.field, road_map)
    game_field = field.field.copy()
    firetruck.draw(game_field)
    wumpus.draw(game_field, PXS_PER_M)
    cv2.imshow('field', game_field)
    cv2.waitKey(1)
    
    # Simulation Parameters
    sim_time = 0
    time_scale = 10.0
    step = int(firetruck.v/road_map.step_size/time_scale)
    w_step = 2.5
    f_path_i = 0
    w_path_i = 0
    f_plan_time = 0
    w_plan_time = 0 
    f_wait = 0
    w_wait = 0
    max_goals = 3 # Number of obstacle groups to attempt to plan to
    while sim_time < 3600 * time_scale:
        game_field = field.field.copy()
        wumpus.score += field.update_burning(time_scale, GRID_SIZE_M) # Update burning locations
        
        '''
        Firetruck control states: DRIVING, IDLE, EXTINGUISHING
        DRIVING: Proceed with path until goal is reached or obstacle is burned
        IDLE: Identify close burning obstacles and generate best path to reach
        EXTINGUISHING: Put out burning obstacles withing 10m
        '''
        if firetruck.state == DRIVING:
            f_path_i += step
            display_path(game_field, [f_segment, f_path], 5)
            if f_path_i>=len(f_segment):
                firetruck.x, firetruck.y, firetruck.heading = f_segment[-1]
                if firetruck.goal not in field.obstacle_states['burning']: # if obstacle is burned, go to idle to find new path
                    firetruck.state = IDLE
                    firetruck.goal = None
                    f_path_i = 0
                elif len(f_path) == 0: # if goal is reached extinguish
                    firetruck.state = EXTINGUISHING  
                    firetruck.goal = None
                    f_path_i = 0
                else:
                    f_path_i = f_path_i - len(f_segment) # proceed to next path segment
                    f_segment = f_path.pop(0)
                    if f_path_i >= len(f_segment):
                        firetruck.x, firetruck.y, firetruck.heading = f_segment[-1]
                        if len(f_path) == 0:
                            firetruck.state = EXTINGUISHING  
                            firetruck.goal = None
                        else:
                            f_segment = f_path.pop(0)
                        f_path_i = 0  
                    else:
                        firetruck.x, firetruck.y, firetruck.heading = f_segment[f_path_i]
            else:
                firetruck.x, firetruck.y, firetruck.heading = f_segment[f_path_i]
        elif firetruck.state == IDLE:
            burning_priority = {}
            for burn_grid_loc in field.obstacle_states['burning'].keys(): # Identify close obstacle groups and prioritize top 3 closest based on Euclidean distance
                burn_m_loc = ((burn_grid_loc[1]+0.5)*GRID_SIZE_M, (burn_grid_loc[0]+0.5)*GRID_SIZE_M)
                dist = math.sqrt((burn_m_loc[0]-firetruck.x)**2 + (burn_m_loc[1]-firetruck.y)**2)
                burn_group_id = field.map_to_obstacle[burn_grid_loc]
                if len(burning_priority) < max_goals:
                    if burn_group_id not in burning_priority:
                        burning_priority[burn_group_id] = dist
                    elif dist < burning_priority[burn_group_id]:
                        burning_priority[burn_group_id] = dist
                else:
                    placed = False
                    for prev_loc,prev_dist in burning_priority.items():
                        if dist<prev_dist:
                            placed = True
                            break
                    burning_priority.popitem()
                    if burn_group_id not in burning_priority:
                        burning_priority[burn_group_id] = dist
                    elif dist < burning_priority[burn_group_id]:
                        burning_priority[burn_group_id] = dist
                    burning_priority = {k: v for k, v in sorted(burning_priority.items(), key=lambda item: item[1])}
            burn_groups = burning_priority.keys()
            prm_goals = [field.obstacle_groups[id] for id in burn_groups] # list of obstacle groups and they're grid locations
            
            if len(prm_goals) != 0:
                plan_start = time.time()
                f_path, updated = road_map.prm_path((firetruck.x, firetruck.y, firetruck.heading), prm_goals, kdtree, 10) # calculate PRM path
                if f_path is not None:
                    f_plan_time = time.time()-plan_start
                    firetruck.time += f_plan_time
                    print("Firetruck plan time: ", f_plan_time)
                    if updated:
                        display_map(field.field, road_map)
                    if len(f_path) == 0: # if burning obstacle is reachable at location, begin extinguishing
                        firetruck.state=EXTINGUISHING
                    else:
                        f_segment = f_path.pop(0) # a path consists of segments between individual poses
                        firetruck.state = DRIVING
                        
        elif firetruck.state == EXTINGUISHING:
            curr_ext_locs = []
            ext_locs = []
            obst_idxs = kdtree.query_ball_point((firetruck.x, firetruck.y), 10) # find obstacles within 10m and update extinguishing time
            for obst_loc_m in kdtree.data[obst_idxs]:
                obst_loc_grid = (int(obst_loc_m[1]/GRID_SIZE_M),int(obst_loc_m[0]/GRID_SIZE_M))
                if obst_loc_grid in field.obstacle_states['burning'] and obst_loc_grid not in curr_ext_locs:
                    curr_ext_locs.append(obst_loc_grid)
                    if obst_loc_grid in firetruck.extinguishing:
                        if firetruck.extinguishing[obst_loc_grid] > 5*time_scale: # set to extinguish if it has been 5s
                            ext_locs.append(obst_loc_grid)
                            firetruck.score += 2
                        else:
                            firetruck.extinguishing[obst_loc_grid] += 1
                    else:
                        firetruck.extinguishing[obst_loc_grid] = 0
            for ext_loc in ext_locs:
                firetruck.extinguishing.pop(ext_loc)
                field.update_obstacle_state(ext_loc, 'burning', 'extinguished', True)      
            if len(curr_ext_locs) == 0:
                firetruck.state = IDLE     
        
        '''
        Wumpus control states: DRIVING, IDLE
        DRIVING: Proceed with path until goal is reached or obstacle is ignited by environment. Ignite obstacle once reached.
        IDLE: Identify close intact obstacles and generate best path to reach
        '''
        if wumpus.state == IDLE:
            closest_intact = None
            closest_intact_dist = 0
            wumpus_grid_loc = (int(wumpus.loc[1]/GRID_SIZE_M), int(wumpus.loc[0]/GRID_SIZE_M))
            for intact_grid_loc in field.obstacle_states['intact']:
                dist = math.sqrt((wumpus_grid_loc[0]-intact_grid_loc[0])**2 + (wumpus_grid_loc[1]-intact_grid_loc[1])**2) # find closest obstacle based on Euclidean dist
                if closest_intact_dist == 0 or dist< closest_intact_dist:
                    neighbors = find_neighbors(intact_grid_loc)
                    blocked = True
                    for neighbor in neighbors:
                        if neighbor[0] < field.small_field.shape[0] and neighbor[1] < field.small_field.shape[0] and not field.small_field[neighbor]:
                            blocked = False
                            break
                    if blocked:
                        print('blocked in')
                    else:
                        closest_intact_dist = dist
                        closest_intact = intact_grid_loc
            if isinstance(closest_intact, tuple):
                w_plan_start = time.time()
                w_path = wumpus.get_path(wumpus_grid_loc, closest_intact) # Plan A* path to location
                w_plan_time = time.time()-w_plan_start
                wumpus.time += w_plan_time
                print("Wumpus plan time: ", w_plan_time)
                wumpus.state = DRIVING
        elif wumpus.state == DRIVING:
            display_path(game_field, w_path[w_path_i:],1)
            if wumpus.goal not in field.obstacle_states['intact']: # stop moving to location if location is no longer intact
                wumpus.state = IDLE
                w_path_i = 0
            else:
                if sim_time%int(w_step*time_scale) == 0: # Iterate through wumpus path only every so often
                    w_path_i += 1
                if w_path_i >= len(w_path):
                    wumpus.loc = w_path[-1]
                    wumpus_grid_loc = (int(wumpus.loc[1]/GRID_SIZE_M), int(wumpus.loc[0]/GRID_SIZE_M))
                    field.update_obstacle_state(wumpus.goal, 'intact', 'burning', True) # ignite obstacle if the location has been reached
                    wumpus.score += 1
                    wumpus.state = IDLE
                    w_path_i = 0
                else:
                    wumpus.loc = w_path[w_path_i]
        
        firetruck.draw(game_field)
        wumpus.draw(game_field, PXS_PER_M)
        
        
        sim_time += 1
        cv2.imshow('field', game_field)
        cv2.waitKey(10)
        if len(field.obstacle_states['intact']) == 0 and len(field.obstacle_states['burning'])==0:
            print("Game Over")
            print(f"Firetruck stats:\nPlanning Time: {round(firetruck.time,2)}, Score: {firetruck.score}")
            print(f"Wumpus stats:\nPlanning Time: {round(wumpus.time,2)}, Score: {wumpus.score}")
            break
        