import sys
import cv2
import pygame
import numpy as np
import time

from obstacle_field import FieldCreator
from path_planner import get_path
from vehicles import Vehicle

MAP = 12
SIZE = 60
FPS = 30

def put_text(win, display_surface):
    # places game winning or losing text on screen
    font = pygame.font.Font(pygame.font.match_font('microsoftsansserif'),32)
    if win:
        text = font.render('YOU WIN!', True, (0,0,0), (0,255,0))
    else:
        text = font.render('GAME OVER!', True, (0,0,0), (255,0,0))
    textRect = text.get_rect()
    textRect.center = (MAP*SIZE/2, MAP*SIZE/2)
    print("YOU WON!")
    display_surface.blit(text,textRect)
    for i in range(20):
        pygame.display.flip()
        time.sleep(0.1)
    
def main():
    pygame.init()
    clock = pygame.time.Clock()
    fc = FieldCreator()
    car_type = 'car'
    bin_field, color_field = fc.createField(0.1, MAP, SIZE, car_type=='truck') # creates a 12x12 grid with 10% fill
    running = True
    # cv2.namedWindow('field', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('field', 640,640)
    cv2.imshow('Parking Lot', color_field)
    U = [0,0]
    origin = (30,60,np.pi/2) if car_type!='truck' else (30,180,np.pi/2)
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
    path_freq = 0.75 # draw location every 0.5s
    path_time = 0
    
    while running:
        try:
            dt = clock.tick(FPS) / 1000.0  # seconds per frame
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