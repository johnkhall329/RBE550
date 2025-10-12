import pygame
import sys
import cv2
import numpy as np
import time

from obstacle_field import FieldCreator
from path_planner import get_path

MAP = 12
SIZE = 60

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
    display_surface = pygame.display.set_mode((MAP*SIZE, MAP*SIZE))
    game_surface = pygame.Surface((MAP*SIZE,MAP*SIZE),pygame.SRCALPHA)
    fc = FieldCreator(game_surface, False)
    field = fc.createField(0.1, MAP) # creates a 12x12 grid with 10% fill
    running = True
    goal_field, goal_loc = place_obj('goal', field, game_surface) # place goal
    hero_group = pygame.sprite.GroupSingle()
    enemy_list = pygame.sprite.Group()
    game_field, hero = place_obj('hero', goal_field) # place hero
    hero.goal_loc = goal_loc
    hero_group.add(hero)
    # cv2.namedWindow('field', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('field', 640,640)
    
    for i in range(10): # places 10 enemies
        game_field, enemy = place_obj('enemy', game_field)
        enemy_list.add(enemy)

    hero.path_find(game_field)
    hero_loc = hero.map_loc
    path_surface = hero.draw_path(hero_loc) # once path is found, draw path
    display_surface.blits([(game_surface,(0,0)),(path_surface,(0,0))])
    hero_group.draw(display_surface)
    enemy_list.draw(display_surface)
    pygame.display.flip()
    while running:
        try:
            path_surface = hero.update(enemy_list,game_field) # move hero and update path
            display_surface.blits([(game_surface,(0,0)),(path_surface,(0,0))])
            for enemy in enemy_list:
                game_field = enemy.update(hero_group,game_field) # move enemies and update game field in case of junk
            collisions = pygame.sprite.groupcollide(enemy_list, enemy_list, False, False)
            for sprite1, collided in collisions.items(): # iterate through enemies to see if any collide together
                for sprite2 in collided:
                    if sprite1 != sprite2: # avoid self collisions
                        game_field=sprite1.teardown(game_field)
                        game_field=sprite2.teardown(game_field)
            hero_group.draw(display_surface)
            enemy_list.draw(display_surface)
            if pygame.sprite.groupcollide(hero_group, enemy_list, False, False): # check if enemy has killed hero
                print("Game Over")
                put_text(False,display_surface)
                running=False
            if hero.map_loc == goal_loc: # check if hero has reached goal
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