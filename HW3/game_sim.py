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
    bin_field, color_field = fc.createField(0.1, MAP, SIZE) # creates a 12x12 grid with 10% fill
    running = True
    # cv2.namedWindow('field', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('field', 640,640)
    cv2.imshow('Parking Lot', color_field)
    U = [0,0]
    car = Vehicle('truck',720,(30,90))
    while running:
        try:
            dt = clock.tick(FPS) / 1000.0  # seconds per frame
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
            game_field, collision_field = car.draw(color_field.copy(),bin_field.copy(),False)
            cv2.imshow('Parking Lot', game_field)
                
        except ValueError as e:
            print(e)
            running=False
        
        # cv2.imshow('field', game_field)
        # cv2.waitKey(1)
        
    sys.exit()

if __name__ == "__main__":
    main()