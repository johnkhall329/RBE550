import pygame
import sys
import cv2
import numpy as np

from obstacle_field import FieldCreator

SIZE = 64
WINDOW = 640

def place_obj(obj,field, display:pygame.Surface, fail = False):
    repeats = 0
    new_field = np.copy(field)
    while True:
        loc = np.random.randint(64,size=2)
        if field[loc[0],loc[1]]:
            repeats +=1
            if fail and repeats >= 5:
                raise Exception("unable to safely place hero") 
        else:
            if obj.lower() == "goal":
                new_field[loc[0],loc[1]] = 1
                pygame.draw.rect(display, (0,255,0), pygame.Rect(loc[1]*(WINDOW/SIZE), loc[0]*(WINDOW/SIZE), (WINDOW/SIZE), (WINDOW/SIZE)))
            elif obj.lower() == "hero":
                new_field[loc[0],loc[1]] = 1
                pygame.draw.circle(display,(0,0,255),[((loc[1]+0.5)*(WINDOW/SIZE)), ((loc[0]+0.5)*(WINDOW/SIZE))], 4,0)
            else:
                enemy_loc = np.array([[((loc[1]+0.5)*(WINDOW/SIZE)), (loc[0]*(WINDOW/SIZE))+1], [(loc[1]*(WINDOW/SIZE))+1, ((loc[0]+1)*(WINDOW/SIZE))-1],[((loc[1]+1)*(WINDOW/SIZE))-1, ((loc[0]+1)*(WINDOW/SIZE))-1]],dtype=np.int16)
                pygame.draw.polygon(display,(255,0,0),enemy_loc,0)
                loc -= [1,1] # shift to center of kernel
                val = np.array([[  0,  50,  100,  50,   0],
                                [ 50, 100, 150, 100,  58],
                                [ 100, 150, 200, 150,  100],
                                [ 50, 100, 150, 100,  50],
                                [  0,  50,  100,  50,   0]], dtype=np.uint8)
                for idx, weight in np.ndenumerate(val):
                    if loc[0]+idx[0]<SIZE and loc[1]+idx[1]<SIZE:
                        if field[loc[0]+idx[0],loc[1]+idx[1]] not in [1,255]:
                            new_field[loc[0]+idx[0],loc[1]+idx[1]] += weight
            return new_field
    
def main():
    pygame.init()

    display_surface = pygame.display.set_mode((WINDOW, WINDOW))
    fc = FieldCreator(display_surface, False)
    field = fc.createField(0.2, 64)

    running = True
    goal_field = place_obj('goal', field, display_surface, False)
    hero_field = place_obj('hero', goal_field, display_surface,False)
    cv2.namedWindow('field', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('field', 720,720)
    enemy_field = place_obj('enemy', hero_field, display_surface,False)
    for i in range(9):
        enemy_field = place_obj('enemy', enemy_field, display_surface,False)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        cv2.imshow('field', enemy_field)
        cv2.waitKey(1)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()