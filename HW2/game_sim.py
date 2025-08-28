import pygame
import sys
import cv2
import numpy as np

from obstacle_field import FieldCreator

SIZE = 64
WINDOW = 640

pygame.init()

display_surface = pygame.display.set_mode((WINDOW, WINDOW))
fc = FieldCreator(display_surface, False)
field = fc.createField(0.2, 64)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()
sys.exit()
