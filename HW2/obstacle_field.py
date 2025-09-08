import numpy as np
import cv2
import pygame
import time

WHITE = (255,255,255)
BLACK = (0,0,0)

class FieldCreator():
    def __init__(self, display:pygame.Surface, view = True):
        self.display = display
        self.view = view

        # Tetronimo shapes
        self.tet_I = np.array([[1,1,1,1]]).reshape((4,1))
        self.tet_L = np.array([[1,0],[1,0],[1,1]])
        self.tet_S = np.array([[1,0],[1,1],[0,1]])
        self.tet_T = np.array([[1,0],[1,1],[1,0]])
        self.tet_list = [self.tet_I, self.tet_L, self.tet_S, self.tet_T]

    def createField(self, coverage, map_size):
        # create blank PyGame display and matrix for path planning
        self.display.fill(WHITE)
        self.field = np.zeros((map_size,map_size),dtype=np.uint8)
        self.cell_size = int(self.display.get_width()/map_size)
        self.placed = 0
        while self.placed < coverage*(map_size**2): # adds tetronimos until it has reached coverage
            # random shape and location
            shape_type = np.random.randint(4)
            loc = (np.random.randint(map_size),np.random.randint(map_size))
            self.place_tet(shape_type, loc, map_size)
            if self.view: 
                pygame.display.flip()
        return self.field
        
    def place_tet(self, shape_type, loc, map_size):
        # take tetronimo and randomly mirror and rotate
        tet = self.tet_list[shape_type]
        if np.random.randint(2): tet = tet.T
        rot = np.random.randint(4)
        tet = np.rot90(tet, rot)

        for idx, val in np.ndenumerate(tet): # if within the field, draw on PyGame display and add to matrix
            if val and loc[0]+idx[0]<map_size and loc[1]+idx[1]<map_size:
                cv2.rectangle(self.field, (loc[1]+idx[1], loc[0]+idx[0]), (loc[1]+idx[1], loc[0]+idx[0]), 255, -1)
                pygame.draw.rect(self.display, BLACK, pygame.Rect((loc[1]+idx[1])*self.cell_size, (loc[0]+idx[0])*self.cell_size, self.cell_size, self.cell_size))
                self.placed += 1


if __name__ == '__main__':
    pygame.init()

    display_surface = pygame.display.set_mode((640, 640))
    fc = FieldCreator(display_surface)
    field = fc.createField(0.2, 64)
    while True:
        pygame.display.flip()
    