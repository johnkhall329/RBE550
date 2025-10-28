import numpy as np
import cv2
import time

INTACT= (0,150,0)
BURNING = (20,150,255)
BURNED = (0,0,102)
EXTINGUISHED = (50,150,150)

class FieldCreator():
    def __init__(self):
        # Tetronimo shapes
        self.tet_I = np.array([[1,1,1,1]]).reshape((4,1))
        self.tet_L = np.array([[1,0],[1,0],[1,1]])
        self.tet_S = np.array([[1,0],[1,1],[0,1]])
        self.tet_T = np.array([[1,0],[1,1],[1,0]])
        self.tet_list = [self.tet_I, self.tet_L, self.tet_S, self.tet_T]
        self.obstacle_states = {"intact": [], "burning": [], "extinguished": [], "burned": []}
        
    def createField(self, coverage, map_size, cell_size):
        self.cell_size = cell_size
        self.field = np.ones((map_size*self.cell_size,map_size*self.cell_size,3),dtype=np.uint8)*255
        self.placed = 0
        self.small_field = np.zeros((map_size,map_size),dtype=np.uint8)
        self.open_spaces = [(0, map_size//2), (map_size-1, map_size//2), (map_size-2, map_size//2)] # spaces for wumpus and truck start
        while self.placed < coverage*(map_size**2): # adds tetronimos until it has reached coverage
            # random shape and location
            shape_type = np.random.randint(4)
            loc = (np.random.randint(map_size),np.random.randint(map_size))
            self.place_tet(shape_type, loc, map_size)
        return self.field, self.small_field, self.obstacle_states
        
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
                self.obstacle_states["intact"].append(pix_loc)
                cv2.rectangle(self.field, (pix_loc[1]*self.cell_size, pix_loc[0]*self.cell_size), ((pix_loc[1]+1)*self.cell_size-1, (pix_loc[0]+1)*self.cell_size-1), INTACT, -1)
                self.placed += 1
                
    def update_field(self, loc, prev_state, new_state):
        try:
            self.obstacle_states[prev_state].remove(loc)
        except ValueError as e:
            print(f'Location: {loc} is not in {prev_state}')
            return
        self.obstacle_states[new_state].append(loc)
        if new_state == "burning":
            cv2.rectangle(self.field, (loc[1]*self.cell_size, loc[0]*self.cell_size), ((loc[1]+1)*self.cell_size-1, (loc[0]+1)*self.cell_size-1), BURNING, -1)
        elif new_state == "extinguished":
            cv2.rectangle(self.field, (loc[1]*self.cell_size, loc[0]*self.cell_size), ((loc[1]+1)*self.cell_size-1, (loc[0]+1)*self.cell_size-1), EXTINGUISHED, -1)
        elif new_state == "burned":
            cv2.rectangle(self.field, (loc[1]*self.cell_size, loc[0]*self.cell_size), ((loc[1]+1)*self.cell_size-1, (loc[0]+1)*self.cell_size-1), BURNED, -1)
        
                
if __name__ == '__main__':
    fc = FieldCreator()
    field, small_field, obstacle_states = fc.createField(0.1, 50, 15)
    print(field.shape)
    cv2.namedWindow('field2', cv2.WINDOW_NORMAL)
    # field2 = cv2.resize(field, (720,720), cv2.)
    # field2 = cv2.bitwise_not(field)
    cv2.imshow('field2', small_field)
    cv2.imshow('field', field)
    cv2.resizeWindow('field2', 720,720)
    while True:
        cv2.waitKey(1)
    