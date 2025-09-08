import pygame
import sys
import cv2
import numpy as np
import time

from obstacle_field import FieldCreator
from path_planner import get_path

MAP = 64
SIZE = 10

class Sprite(pygame.sprite.Sprite):
    def __init__(self, role, map_loc):
        super().__init__()
        self.role = role
        self.map_loc = map_loc
        self.screen_loc = (self.map_loc[1]*SIZE, self.map_loc[0]*SIZE)
        self.teleports = 0
        self.image = pygame.Surface([SIZE, SIZE], pygame.SRCALPHA)
        self.active = True
        if role == "hero":
            pygame.draw.circle(self.image,(0,0,255),(0.5*SIZE,0.5*SIZE), 4,0)
        elif role == "enemy":
            self.kernel = np.array([[  0,  25,  50,  25,   0],
                                    [ 25, 50, 75, 50,  25],
                                    [ 50, 75, 100, 75,  50],
                                    [ 25, 50, 75, 50,  25],
                                    [  0,  25,  50,  25,   0]])
            enemy_loc = np.array([((0.5*SIZE), 1), (1, SIZE-1),(SIZE-1, SIZE-1)], dtype=np.int16)
            pygame.draw.polygon(self.image,(255,0,0),enemy_loc,0)

        self.rect = self.image.get_rect()
        self.rect.topleft=self.screen_loc
    
    def update(self, sprites:pygame.sprite.Group, field:np.ndarray):
        if self.role == 'hero':
            if len(self.path) > 0:
                next_node = self.path.pop()
                enemy_dist = 500
                for enemy in sprites:
                    if enemy.active:
                        enemy_dist = min(enemy_dist, (abs(enemy.map_loc[0]-self.map_loc[0]) + abs(enemy.map_loc[1]-self.map_loc[1])))
                if enemy_dist <= 3:
                    self.path_find(field,new_loc=True)
                    next_node = self.path.pop()
                elif enemy_dist <= 10 or field[next_node] == 255:            
                    self.path_find(field)
                    next_node = self.path.pop()
                surface = self.draw_path(next_node)
                self.map_loc = next_node
                self.screen_loc = (self.map_loc[1]*SIZE, self.map_loc[0]*SIZE)
                self.rect.topleft=self.screen_loc
            else: surface = pygame.Surface((MAP*SIZE,MAP*SIZE), pygame.SRCALPHA)
            return surface
        
        elif self.role == 'enemy':
            if self.active:
                hero = sprites.sprites()[0]
                x_dist = hero.rect.x - self.rect.x
                y_dist = hero.rect.y - self.rect.y
                if abs(x_dist) >= abs(y_dist): new_loc = (self.map_loc[0], self.map_loc[1] + np.sign(x_dist))
                else: new_loc = (self.map_loc[0] + np.sign(y_dist), self.map_loc[1])
                if field[new_loc] == 255:
                    field = self.teardown(field)
                else:
                    for idx, weight in np.ndenumerate(self.kernel):
                        idx = (idx[0]-2,idx[1]-2)
                        if self.map_loc[0]+idx[0]<MAP and self.map_loc[1]+idx[1]<MAP:
                            if field[self.map_loc[0]+idx[0],self.map_loc[1]+idx[1]] not in [1,255]:
                                field[self.map_loc[0]+idx[0],self.map_loc[1]+idx[1]] -= weight
                        if new_loc[0]+idx[0]<MAP and new_loc[1]+idx[1]<MAP:
                            if field[new_loc[0]+idx[0],new_loc[1]+idx[1]] not in [1,255]:
                                field[new_loc[0]+idx[0],new_loc[1]+idx[1]] += weight
                                if  field[new_loc[0]+idx[0],new_loc[1]+idx[1]] > 255: field[new_loc[0]+idx[0],new_loc[1]+idx[1]] = 255
                    self.map_loc = new_loc
                    self.screen_loc = (self.map_loc[1]*SIZE, self.map_loc[0]*SIZE)
                    self.rect.topleft=self.screen_loc
            return field
    
    def teardown(self,field):
        pygame.draw.rect(self.image,(0,0,0),pygame.Rect(0,0,SIZE,SIZE))
        self.active = False
        field[self.map_loc] = 255
        for idx, weight in np.ndenumerate(self.kernel):
            idx = (idx[0]-2,idx[1]-2)
            if self.map_loc[0]+idx[0]<MAP and self.map_loc[1]+idx[1]<MAP:
                if field[self.map_loc[0]+idx[0],self.map_loc[1]+idx[1]] not in [1,255]:
                    field[self.map_loc[0]+idx[0],self.map_loc[1]+idx[1]] -= weight
        return field

    def path_find(self, field, goal_loc=None, curr_loc=None, new_loc=False):
        try:
            if new_loc:
                while True:
                    if self.teleports >= 5: break
                    loc = np.random.randint(64,size=2)
                    if field[loc[0],loc[1]] not in [1,255]:
                        curr_loc = tuple(loc)
                        self.teleports += 1
                        break
            if not goal_loc: goal_loc = self.goal_loc
            if not curr_loc: curr_loc = self.map_loc
            self.path = get_path(field, goal_loc, curr_loc)
        except ValueError as e:
            if self.teleports >= 5: raise(e)
            self.path_find(field,goal_loc,curr_loc=True)
    
    def draw_path(self,curr_loc):
        surface = pygame.Surface((MAP*SIZE,MAP*SIZE), pygame.SRCALPHA)
        prev_node = self.goal_loc
        for i in range(len(self.path)):
            node = self.path[i]
            pygame.draw.line(surface, (255,0,0), ((prev_node[1]+0.5)*SIZE, (prev_node[0]+0.5)*SIZE), ((node[1]+0.5)*SIZE, (node[0]+0.5)*SIZE))
            prev_node = node
        pygame.draw.line(surface, (255,0,0), ((prev_node[1]+0.5)*SIZE, (prev_node[0]+0.5)*SIZE), ((curr_loc[1]+0.5)*SIZE, (curr_loc[0]+0.5)*SIZE))
        return surface

def place_obj(obj,field, display:pygame.Surface = None):
    new_field = np.copy(field)
    while True:
        loc = np.random.randint(MAP,size=2)
        if not field[loc[0],loc[1]]:
            if obj.lower() == "goal":
                new_field[loc[0],loc[1]] = 1
                pygame.draw.rect(display, (0,255,0), pygame.Rect(loc[1]*SIZE, loc[0]*SIZE, SIZE, SIZE))
                sprite = tuple(loc)
            elif obj.lower() == "hero":
                sprite = Sprite("hero", tuple(loc))
            else:
                sprite = Sprite("enemy", tuple(loc))
                loc -= [2,2] # shift to center of kernel
                for idx, weight in np.ndenumerate(sprite.kernel):
                    if loc[0]+idx[0]<MAP and loc[1]+idx[1]<MAP:
                        if field[loc[0]+idx[0],loc[1]+idx[1]] not in [1,255]:
                            new_field[loc[0]+idx[0],loc[1]+idx[1]] += weight
                            if new_field[loc[0]+idx[0],loc[1]+idx[1]] > 255: new_field[loc[0]+idx[0],loc[1]+idx[1]] = 255
            return new_field, sprite

def put_text(win, display_surface):
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
    display_surface = pygame.display.set_mode((MAP*SIZE, MAP*SIZE))
    game_surface = pygame.Surface((MAP*SIZE,MAP*SIZE),pygame.SRCALPHA)
    fc = FieldCreator(game_surface, False)
    field = fc.createField(0.2, 64)
    running = True
    goal_field, goal_loc = place_obj('goal', field, game_surface)
    hero_group = pygame.sprite.GroupSingle()
    enemy_list = pygame.sprite.Group()
    game_field, hero = place_obj('hero', goal_field)
    hero.goal_loc = goal_loc
    hero_group.add(hero)
    # cv2.namedWindow('field', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('field', 640,640)
    
    for i in range(10):
        game_field, enemy = place_obj('enemy', game_field)
        enemy_list.add(enemy)

    hero.path_find(game_field)
    hero_loc = hero.map_loc
    path_surface = hero.draw_path(hero_loc)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        try:
            path_surface = hero.update(enemy_list,game_field)
            display_surface.blits([(game_surface,(0,0)),(path_surface,(0,0))])
            for enemy in enemy_list:
                game_field = enemy.update(hero_group,game_field)
            collisions = pygame.sprite.groupcollide(enemy_list, enemy_list, False, False)
            for sprite1, collided in collisions.items():
                for sprite2 in collided:
                    if sprite1 != sprite2:
                        game_field=sprite1.teardown(game_field)
                        game_field=sprite2.teardown(game_field)
            hero_group.draw(display_surface)
            enemy_list.draw(display_surface)
            if pygame.sprite.groupcollide(hero_group, enemy_list, False, False):
                print("Game Over")
                put_text(False,display_surface)
                running=False
            if hero.map_loc == goal_loc: 
                put_text(True,display_surface)
                running=False
                
        except ValueError as e:
            print(e)
            put_text(False,display_surface)
            running=False
        
        # cv2.imshow('field', game_field)
        # cv2.waitKey(1)
        
        pygame.display.flip()
        time.sleep(0.25)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()