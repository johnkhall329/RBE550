import pygame
import sys
import cv2
import numpy as np
import time

from obstacle_field import FieldCreator
from path_planner import get_path

SIZE = 64
WINDOW = 640

class Sprite(pygame.sprite.Sprite):
    def __init__(self, role, map_loc):
        super().__init__()
        self.role = role
        self.map_loc = map_loc
        self.screen_loc = (self.map_loc[1]*(WINDOW/SIZE), self.map_loc[0]*(WINDOW/SIZE))
        self.teleports = 0
        self.image = pygame.Surface([WINDOW/SIZE,WINDOW/SIZE], pygame.SRCALPHA)
        if role == "hero":
            pygame.draw.circle(self.image,(0,0,255),(0.5*WINDOW/SIZE,0.5*WINDOW/SIZE), 4,0)
        elif role == "enemy":
            enemy_loc = np.array([((0.5*WINDOW/SIZE), 1), (1, WINDOW/SIZE-1),(WINDOW/SIZE-1, WINDOW/SIZE-1)], dtype=np.int16)
            pygame.draw.polygon(self.image,(255,0,0),enemy_loc,0)

        self.rect = self.image.get_rect()
        self.rect.topleft=self.screen_loc
    
    def update(self, enemies:pygame.sprite.Group):
        if self.role == 'hero':
            if self.progress < len(self.path):
                next_node = self.path[self.progress]
                self.progress += 1
                self.map_loc = next_node
            self.screen_loc = (self.map_loc[1]*(WINDOW/SIZE), self.map_loc[0]*(WINDOW/SIZE))
            self.rect.topleft=self.screen_loc
            surface = self.draw_path()
            return surface
    
    def path_find(self, field, goal_loc):
        try:
            self.path = get_path(field, goal_loc, self.map_loc)
            self.progress = 0
            return self.draw_path()
        except Exception as e:
            print(e)
    
    def draw_path(self):
        surface = pygame.Surface((WINDOW,WINDOW), pygame.SRCALPHA)
        for i in range(self.progress, len(self.path)):
            node = self.path[i]
            prev_node = self.path[i-1]
            pygame.draw.line(surface, (255,0,0), ((prev_node[1]+0.5)*WINDOW/SIZE, (prev_node[0]+0.5)*WINDOW/SIZE), ((node[1]+0.5)*WINDOW/SIZE, (node[0]+0.5)*WINDOW/SIZE))
        return surface


def place_obj(obj,field, display:pygame.Surface = None):
    new_field = np.copy(field)
    while True:
        loc = np.random.randint(64,size=2)
        if not field[loc[0],loc[1]]:
            if obj.lower() == "goal":
                new_field[loc[0],loc[1]] = 1
                pygame.draw.rect(display, (0,255,0), pygame.Rect(loc[1]*(WINDOW/SIZE), loc[0]*(WINDOW/SIZE), (WINDOW/SIZE), (WINDOW/SIZE)))
                sprite = tuple(loc)
            elif obj.lower() == "hero":
                sprite = Sprite("hero", tuple(loc))
            else:
                sprite = Sprite("enemy", tuple(loc))
                loc -= [2,2] # shift to center of kernel
                val = np.array([[  0,  25,  50,  25,   0],
                                [ 25, 50, 75, 50,  25],
                                [ 100, 75, 100, 75,  100],
                                [ 25, 50, 75, 50,  25],
                                [  0,  25,  50,  25,   0]], dtype=np.uint8)
                for idx, weight in np.ndenumerate(val):
                    if loc[0]+idx[0]<SIZE and loc[1]+idx[1]<SIZE:
                        if field[loc[0]+idx[0],loc[1]+idx[1]] not in [1,255]:
                            new_field[loc[0]+idx[0],loc[1]+idx[1]] += weight
            return new_field, sprite
    
def main():
    pygame.init()

    display_surface = pygame.display.set_mode((WINDOW, WINDOW))
    game_surface = pygame.Surface((640,640),pygame.SRCALPHA)
    fc = FieldCreator(game_surface, False)
    field = fc.createField(0.2, 64)
    running = True
    goal_field, goal_loc = place_obj('goal', field, game_surface)
    hero_group = pygame.sprite.GroupSingle()
    enemy_list = pygame.sprite.Group()
    game_field, hero = place_obj('hero', goal_field)
    hero_group.add(hero)
    # cv2.namedWindow('field', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('field', 720,720)
    
    for i in range(10):
        game_field, enemy = place_obj('enemy', game_field)
        enemy_list.add(enemy)

    path_surface = hero.path_find(game_field, goal_loc)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # cv2.imshow('field', game_field)
        # cv2.waitKey(1)
        
        path_surface = hero.update(enemy_list)
        display_surface.blits([(game_surface,(0,0)),(path_surface,(0,0))])
        enemy_list.update(enemy_list)
        hero_group.draw(display_surface)
        enemy_list.draw(display_surface)
        
        
        pygame.display.flip()
        time.sleep(0.5)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()