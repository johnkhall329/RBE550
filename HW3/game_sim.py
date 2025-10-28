import sys
import cv2
import pygame
import numpy as np
import time
import math

from unconstrained import Unconstrained
from obstacle_field import FieldCreator
from hybrid_a_star import HybridAStar
from vehicles import Vehicle

MAP = 12
SIZE = 90
FPS = 30
VEL_SCALE = 4
RESOLUTION = 15
ANG_RESOLUTION = np.pi/8

DELIVERY = 'delivery'
CAR = 'car'
TRUCK = 'truck'

def check_map(map):
    '''
    Check if path is even viable from start to goal using unconstrained A*
    '''
    twod_astar = Unconstrained((11,9),map, False)
    return twod_astar.get_unconstrained_path((0,0),step_size=1) != 1e9

def main():
    pygame.init()
    
    fc = FieldCreator()
    car_type = CAR
    bin_field, color_field, small_field = fc.createField(0.1, MAP, SIZE, car_type==TRUCK) # creates a 12x12 grid with 10% fill
    running = True
    
    U = [0,0]
    truck_offset = 180
    origin = (45,90, np.pi/2) if car_type!=TRUCK else (45,90+truck_offset, np.pi/2) # set start and goal location depending on car type
    goal = (810,1035) if car_type!=TRUCK else (810-truck_offset/2,1035) 
    car = Vehicle(car_type,MAP*SIZE,origin)
    
    # Check if path is even possible
    if not check_map(small_field):
        print("invalid map")
        return
    
    # Draw start and goal locations
    if car_type == 'truck': 
        all_points = np.vstack([car.box.points(), car.trailer_box.points()])
        big_rect = cv2.minAreaRect(all_points)
        points = np.int32(cv2.boxPoints(big_rect))
        cv2.drawContours(color_field, [points],0,(255,0,0),2)
        goal_rect = (goal, big_rect[1], 0)
        goal_points = np.int32(cv2.boxPoints(goal_rect))
        cv2.drawContours(color_field, [goal_points],0,(0,255,0),2)
    else:
        cv2.drawContours(color_field, [car.box.points().astype(np.int32).reshape((-1, 1, 2))],0,(255,0,0),2)
        goal_rect = cv2.RotatedRect(goal, (156,54), 0)
        cv2.drawContours(color_field, [goal_rect.points().astype(np.int32).reshape((-1, 1, 2))],0,(0,255,0),2)
        
    cv2.circle(color_field,(int(car.x),int(car.y)), 1, (255,0,0),-1)
    path_freq = 1 # draw location every 1s
    path_time = 0
        
    start_loc = (car.x, car.y, car.heading, 'S') if car_type!=TRUCK else (car.x, car.y, car.heading, car.trailer_heading, 'S')
    goal_loc1 = (goal[0]+car.wheelbase/2, goal[1], np.pi) if car_type!=TRUCK else (goal[0] - truck_offset/2 + (car.wheelbase)/2, goal[1], np.pi, np.pi) # set two goal locations at different directions
    goal_loc2 = (goal[0]-car.wheelbase/2, goal[1], 0) if car_type!=TRUCK else (goal[0] + truck_offset/2 - (car.wheelbase)/2, goal[1], 0, 0)
    cv2.circle(color_field,(int(goal_loc1[0]),int(goal_loc1[1])), 1, (0,255,0),-1)
    cv2.circle(color_field,(int(goal_loc2[0]),int(goal_loc2[1])), 1, (0,255,0),-1)
    cv2.imshow('Parking Lot', color_field)
    cv2.waitKey(1)
    
    atime = time.time()
    hybrid_astar = HybridAStar(bin_field, car, RESOLUTION, ANG_RESOLUTION)
    path, rs_path = hybrid_astar.hybrid_a_star_path(start_loc, [goal_loc1, goal_loc2])
    astar_length = len(path)
    total_path = path+rs_path
    print(time.time()-atime)
    
    # cm = color_field.copy()
    ARROW_LENGTH = 5
    for i,loc in enumerate(total_path): # Iterate through and draw path
        if car.type != 'truck':
            x,y,phi,dir = loc
        else:
            x,y,phi,tphi,dir = loc
        center = (int(x), int(y))

        # Draw the center point
        if i < astar_length:
            color = (255, 0, 0)
        else: 
            color = (255, 255, 0)
        cv2.circle(color_field, center, 3, color, -1)

        # Compute arrow direction (positive heading = CCW)
        dx = ARROW_LENGTH * math.cos(phi)
        dy = ARROW_LENGTH * math.sin(phi)
        tip = (int(x + dx), int(y + dy))

        # Draw heading arrow
        cv2.arrowedLine(color_field, center, tip, (0, 0, 255), 2, tipLength=0.4)
        # cv2.putText(color_field, dir, (center[0]+5, center[1]-5), cv2.FONT_HERSHEY_COMPLEX,0.5, color)
        cv2.imshow('Parking Lot', color_field)
        cv2.waitKey(1)

    # Prepare motion interpolation
    next_node = start_loc
    prev_node = next_node
    thresh = 1
    ang_thresh = 0.1
    time.sleep(1)
    U = [0,0]
    clock = pygame.time.Clock()
    step_time = 0
    run_path = False
    while running:
        try:
            dt = clock.tick(FPS) / 1000.0  # seconds per frame    
            path_time += dt
            step_time += dt
            key = cv2.waitKey(1)
            
            # For manual vehicle control
            if key==ord('w'):
                U[1] += 2
            if key==ord('s'):
                U[1] -= 2
            if key==ord('a'):
                U[0] = max(U[0]-np.deg2rad(5),-np.pi/3)
            if key==ord('d'):
                U[0] = min(U[0]+np.deg2rad(5),np.pi/3)
            if key==13:
                run_path = True
            if key==ord('p'):
                np.save(f'HW3/{int(time.time())}_field', color_field)
                np.save(f'HW3/{int(time.time())}_path', total_path)

            if not run_path:
                game_field, color_field = car.draw(color_field.copy(), color_field, True)
                continue
            
            # If the vehicle reaches the next node in the path, update to next
            if abs(car.x-next_node[0]) < thresh and abs(car.y-next_node[1]) < thresh and abs(car.heading-next_node[2]) < ang_thresh:
                prev_node = next_node
                if len(total_path) == 0: # Handle completion
                    print('completed')
                    U=[0,0]
                    time.sleep(2)
                    continue
                next_node = total_path.pop(0)
                
                # In case next node is same as previous (Reeds-Shepp path output)
                if abs(car.x-next_node[0]) < thresh and abs(car.y-next_node[1]) < thresh and abs(car.heading-next_node[2]) < ang_thresh:
                    # print('here again')
                    prev_node = next_node
                    next_node = total_path.pop(0)
            
                car.x, car.y, car.heading = prev_node[:3]
                if car_type == TRUCK: car.trailer_heading = prev_node[3]
                step_time = 0  
                        
            # In case interpolation takes longer than it should
            if step_time > 1/VEL_SCALE:
                car.x, car.y, car.heading = next_node[:3]
                if car_type == TRUCK: car.trailer_heading = next_node[3]
                if len(total_path) == 0: # Handle completion
                    print('completed')
                    U=[0,0]
                    time.sleep(2)
                    continue
                next_node = total_path.pop(0)
                step_time = 0
                                
            dir = next_node[-1]
            # Set velocity
            if dir == 'STOP':
                U[1] = 0
            else:
                U[1] = -RESOLUTION*VEL_SCALE if 'B' in dir else RESOLUTION*VEL_SCALE
            
            # Set steering angle of car or truck
            if car_type != 'delivery':
                if dir == 'S' or dir == 'B' or dir == 'STOP':
                    U[0] = 0
                elif dir in ['R', 'BR']:
                    U[0] = math.atan(ANG_RESOLUTION*car.wheelbase/RESOLUTION)
                else:
                    U[0] = -math.atan(ANG_RESOLUTION*car.wheelbase/RESOLUTION)
            else: # Delivery robot angular velocity
                if dir == 'S' or dir == 'B' or dir == 'STOP':
                    U[0] = 0
                elif dir in ['R', 'BL']:
                    U[0] = ANG_RESOLUTION
                    # U[1] = 0
                else:
                    U[0] = -ANG_RESOLUTION
                    # U[1] = 0
                    
            car.update(U,dt) # Update car position
            
            # Stamp location
            if path_time > path_freq:
                path_time = 0
                game_field, color_field = car.draw(color_field.copy(), color_field, True)
            else:
                game_field, color_field = car.draw(color_field.copy(), color_field, False)
                
            cv2.imshow('Parking Lot', game_field)
            
        except ValueError as e:
            print(e)
            running=False
        
    sys.exit()

if __name__ == "__main__":
    main()