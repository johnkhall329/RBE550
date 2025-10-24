import sys
import cv2
import pygame
import numpy as np
import time
import math

from obstacle_field import FieldCreator
from hybrid_a_star import hybrid_a_star_path, round_node
from vehicles import Vehicle

MAP = 12
SIZE = 60
FPS = 30
VEL_SCALE = 1
VELOCITY = 30 * VEL_SCALE
ANG_VELOCITY = np.pi/8 * VEL_SCALE


def main():
    pygame.init()
    
    fc = FieldCreator()
    car_type = 'delivery'
    bin_field, color_field = fc.createField(0.1, MAP, SIZE, car_type=='truck') # creates a 12x12 grid with 10% fill
    running = True
    # cv2.namedWindow('field', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('field', 640,640)
    
    U = [0,0]
    origin = (30,60) if car_type!='truck' else (30,180,np.pi/2)
    goal = (540,690) if car_type!='truck' else (480,690) 
    car = Vehicle(car_type,MAP*SIZE,origin)
    
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
        goal_rect = cv2.RotatedRect(goal, (104,36), 0)
        cv2.drawContours(color_field, [goal_rect.points().astype(np.int32).reshape((-1, 1, 2))],0,(0,255,0),2)
        
    cv2.circle(color_field,(int(car.x),int(car.y)), 1, (255,0,0),-1)
    path_freq = 0.5 # draw location every 0.5s
    path_time = 0
    cv2.imshow('Parking Lot', color_field)
    cv2.waitKey(1)
    atime = time.time()
    path, rs_path = hybrid_a_star_path((origin[0],origin[1]-car.wheelbase/2, np.pi/2, 'S'), [(goal[0]+car.wheelbase/2, goal[1], np.pi),(goal[0]-car.wheelbase/2, goal[1], 0)], bin_field, car)
    astar_length = len(path)
    total_path = path+rs_path
    
    print(time.time()-atime)
    # cm = color_field.copy()
    ARROW_LENGTH = 5
    smoothed_path = []
    for i,loc in enumerate(total_path):
        x, y, phi, dir = loc
        smoothed_path.append([x,y,phi])
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
    # np.save('hybrid_astar_path.npy', np.array(smoothed_path, dtype=np.float32))
    next_x, next_y, next_phi, dir = (origin[0],origin[1]-car.wheelbase/2, np.pi/2, 'S')
    prev_x, prev_y, prev_phi, prev_dir = next_x, next_y, next_phi, dir
    thresh = 1
    ang_thresh = 0.1
    # dir = 'STOP'
    time.sleep(1)
    U = [0,0]
    clock = pygame.time.Clock()
    while running:
        try:
            dt = clock.tick(FPS) / 1000.0  # seconds per frame    
            # dt = 1.0/FPS
            path_time += dt
            key = cv2.waitKey(1)
            
            if key==ord('w'):
                U[1] += 2
            if key==ord('s'):
                U[1] -= 2
            if key==ord('a'):
                U[0] = max(U[0]-np.deg2rad(5),-np.pi/3)
            if key==ord('d'):
                U[0] = min(U[0]+np.deg2rad(5),np.pi/3)
            
            if abs(car.x-next_x) < thresh and abs(car.y-next_y) < thresh and abs(car.heading-next_phi) < ang_thresh:
                print('here')
                prev_x, prev_y, prev_phi, prev_dir = next_x, next_y, next_phi, dir
                if len(total_path) == 0:
                    print('completed')
                    U=[0,0]
                    time.sleep(2)
                    continue
                next_x, next_y, next_phi, dir = total_path.pop(0)
                # print(dir)
                if abs(car.x-next_x) < thresh and abs(car.y-next_y) < thresh and abs(car.heading-next_phi) < ang_thresh:
                    # print('here again')
                    prev_x, prev_y, prev_phi, prev_dir = next_x, next_y, next_phi, dir
                    next_x, next_y, next_phi, dir = total_path.pop(0)
            
                car.x, car.y, car.heading = prev_x, prev_y, prev_phi
                # change = math.atan2(next_y-prev_y, next_x-prev_x) 
            
            
            if dir == 'STOP':
                U[1] = 0
            else:
                U[1] = -VELOCITY if 'B' in dir else VELOCITY
            
            if car_type != 'delivery':
                if dir == 'S' or dir == 'B' or dir == 'STOP':
                    U[0] = 0
                elif dir in ['R', 'BR']:
                    U[0] = math.atan(ANG_VELOCITY*car.wheelbase/VELOCITY)
                else:
                    U[0] = -math.atan(ANG_VELOCITY*car.wheelbase/VELOCITY)
            else:
                if dir == 'S' or dir == 'B' or dir == 'STOP':
                    U[0] = 0
                elif dir in ['R', 'BL']:
                    U[0] = ANG_VELOCITY
                else:
                    U[0] = -ANG_VELOCITY
            # print(U)
            car.update(U,dt)
            if path_time > path_freq:
                path_time = 0
                game_field, color_field = car.draw(color_field.copy(), color_field, True)
            else:
                game_field, color_field = car.draw(color_field.copy(), color_field, False)
                
            cv2.imshow('Parking Lot', game_field)
            
        except ValueError as e:
            print(e)
            running=False
        
        # cv2.imshow('field', game_field)
        # cv2.waitKey(1)
        
    sys.exit()

if __name__ == "__main__":
    main()