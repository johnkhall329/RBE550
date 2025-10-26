import numpy as np
import cv2
import time

WHITE = (255,255,255)
BLACK = (0,0,0)

class FieldCreator():
    def __init__(self):
        # Tetronimo shapes
        self.tet_I = np.array([[1,1,1,1]]).reshape((4,1))
        self.tet_L = np.array([[1,0],[1,0],[1,1]])
        self.tet_S = np.array([[1,0],[1,1],[0,1]])
        self.tet_T = np.array([[1,0],[1,1],[1,0]])
        self.block = np.array([[1]])
        self.tet_list = [self.tet_I, self.tet_L, self.tet_S, self.tet_T, self.block]
        
    def createField(self, coverage, map_size, cell_size, truck = False):
        self.cell_size = cell_size
        self.small_field = np.zeros((map_size,map_size),dtype=np.uint8)
        self.field = np.zeros((map_size*self.cell_size,map_size*self.cell_size),dtype=np.uint8)
        self.placed = 0
        self.open_spaces = [(0,0),(1,0),(0,1),(1,1),(map_size-1, map_size-3),(map_size-1, map_size-4)] # spaces for start, goal, and adjacent
        if truck: 
            self.open_spaces.append((map_size-1, map_size-5))
            self.open_spaces.append((map_size-1, map_size-6))
            self.open_spaces.append((2,0))
            self.open_spaces.append((3,0))
            self.place_tet(4, (map_size-1, map_size-2), map_size) # place obstacles near parking spot
            self.place_tet(4, (map_size-1, map_size-7), map_size)
        else:
            self.place_tet(4, (map_size-1, map_size-2), map_size)
            self.place_tet(4, (map_size-1, map_size-5), map_size)
        while self.placed < coverage*(map_size**2): # adds tetronimos until it has reached coverage
            # random shape and location
            shape_type = np.random.randint(4)
            loc = (np.random.randint(map_size),np.random.randint(map_size))
            self.place_tet(shape_type, loc, map_size)
        return self.field, cv2.cvtColor(cv2.bitwise_not(self.field), cv2.COLOR_GRAY2BGR), self.small_field
        
    def place_tet(self, shape_type, loc, map_size):
        # take tetronimo and randomly mirror and rotate
        tet = self.tet_list[shape_type]
        if np.random.randint(2): tet = tet.T
        rot = np.random.randint(4)
        tet = np.rot90(tet, rot)

        for idx, val in np.ndenumerate(tet): # if within the field, draw on PyGame display and add to matrix
            pix_loc = (loc[0]+idx[0], loc[1]+idx[1])
            if val and pix_loc[0]<map_size and pix_loc[1]<map_size and pix_loc not in self.open_spaces:
                self.small_field[pix_loc] = 255
                cv2.rectangle(self.field, (pix_loc[1]*self.cell_size, pix_loc[0]*self.cell_size), ((pix_loc[1]+1)*self.cell_size-1, (pix_loc[0]+1)*self.cell_size-1), 255, -1)
                self.placed += 1


if __name__ == '__main__':
    fc = FieldCreator()
    field, field2, field3 = fc.createField(0.1, 12, 60, True)
    cv2.namedWindow('field3', cv2.WINDOW_NORMAL)
    # field2 = cv2.resize(field, (720,720), cv2.)
    field2 = cv2.bitwise_not(field)
    cv2.imshow('field3', field3)
    cv2.imshow('field', field)
    cv2.resizeWindow('field3', 720,720)
    while True:
        cv2.waitKey(1)
    