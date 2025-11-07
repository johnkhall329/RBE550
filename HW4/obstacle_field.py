import numpy as np
import cv2
import time
import math

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
        self.obstacle_states = {"intact": [], "burning": {}, "extinguished": [], "burned": []}
        self.obstacle_groups = []
        self.map_to_obstacle = {}
        
        
    def createField(self, coverage, map_size, cell_size, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
        else: self.rng = np.random.default_rng(seed=rng)
        self.cell_size = cell_size
        self.field = np.ones((map_size*self.cell_size,map_size*self.cell_size,3),dtype=np.uint8)*255
        self.placed = 0
        self.total_obst = 0
        self.small_field = np.zeros((map_size,map_size),dtype=np.uint8)
        self.open_spaces = [(0, map_size//2), (map_size-1, map_size//2), (map_size-2, map_size//2)] # spaces for wumpus and truck start
        while self.placed < coverage*(map_size**2): # adds tetronimos until it has reached coverage
            # random shape and location
            shape_type = self.rng.integers(4)
            loc = (self.rng.integers(map_size),self.rng.integers(map_size))
            self.place_tet(shape_type, loc, map_size)
        return self.field, self.small_field, self.obstacle_states
        
    def place_tet(self, shape_type, loc, map_size):
        # take tetronimo and randomly mirror and rotate
        tet = self.tet_list[shape_type]
        if self.rng.integers(2): tet = tet.T
        rot = self.rng.integers(4)
        tet = np.rot90(tet, rot)
        self.obstacle_groups.append([])
        for idx, val in np.ndenumerate(tet): # if within the field, draw on PyGame display and add to matrix
            pix_loc = (loc[0]+idx[0], loc[1]+idx[1])
            if val and pix_loc[0]<map_size and pix_loc[1]<map_size and pix_loc not in self.open_spaces and not self.small_field[pix_loc]:
                self.small_field[pix_loc] = 255
                self.obstacle_states["intact"].append(pix_loc)
                self.map_to_obstacle[pix_loc] = self.total_obst
                self.obstacle_groups[self.total_obst].append(pix_loc)
                cv2.rectangle(self.field, (pix_loc[1]*self.cell_size, pix_loc[0]*self.cell_size), ((pix_loc[1]+1)*self.cell_size-1, (pix_loc[0]+1)*self.cell_size-1), INTACT, -1)
                self.placed += 1
        self.total_obst += 1
                
    def update_obstacle_state(self, loc, prev_state, new_state, group=False):
        if group:
                group_id = self.map_to_obstacle[loc]
                for obst_loc in self.obstacle_groups[group_id]:
                    self.update_obstacle_state(obst_loc, prev_state, new_state, False)
                return
        try:
            if prev_state != 'burning':
                self.obstacle_states[prev_state].remove(loc)
            else:
                self.obstacle_states[prev_state].pop(loc)
                
        except (ValueError, KeyError) as e:
            print(f'Location: {loc} is not in {prev_state}')
            return
        if new_state == "burning":
            self.obstacle_states[new_state][loc] = 0
            cv2.rectangle(self.field, (loc[1]*self.cell_size, loc[0]*self.cell_size), ((loc[1]+1)*self.cell_size-1, (loc[0]+1)*self.cell_size-1), BURNING, -1)
        elif new_state == "extinguished":
            self.obstacle_states[new_state].append(loc)
            cv2.rectangle(self.field, (loc[1]*self.cell_size, loc[0]*self.cell_size), ((loc[1]+1)*self.cell_size-1, (loc[0]+1)*self.cell_size-1), EXTINGUISHED, -1)
        elif new_state == "burned":
            self.obstacle_states[new_state].append(loc)
            cv2.rectangle(self.field, (loc[1]*self.cell_size, loc[0]*self.cell_size), ((loc[1]+1)*self.cell_size-1, (loc[0]+1)*self.cell_size-1), BURNED, -1)
    
    def update_burning(self):
        burned_groups = []
        for loc, time in self.obstacle_states['burning'].items():
            if time > 100:
                group_id = self.map_to_obstacle[loc]
                if group_id not in burned_groups: burned_groups.append(group_id)
            else:
                self.obstacle_states['burning'][loc] += 1
        for b_obst in burned_groups:
            b_loc = self.obstacle_groups[b_obst][0]
            self.update_obstacle_state(b_loc, 'burning', 'burned', True)

            close_intacts = []
            for intact_obst in self.obstacle_states['intact']:
                if math.sqrt((intact_obst[0]-b_loc[0])**2 + (intact_obst[1]-b_loc[1])**2) <=12:
                    close_intacts.append(intact_obst)
            
            if len(close_intacts) != 0:
                intact_groups = []
                for b_loc in self.obstacle_groups[b_obst]:
                    for close_intact in close_intacts:
                        if math.sqrt((close_intact[0]-b_loc[0])**2 + (close_intact[1]-b_loc[1])**2) <=6:
                            intact_id = self.map_to_obstacle[close_intact]
                            if intact_id not in intact_groups: 
                                intact_groups.append(intact_id)
                                self.update_obstacle_state(close_intact, 'intact', 'burning', True)
                        
                    
        
        
            
                
if __name__ == '__main__':
    fc = FieldCreator()
    rng = np.random.default_rng()
    field, small_field, obstacle_states = fc.createField(0.005, 50, 15, rng)
    print(fc.map_to_obstacle)
    cv2.namedWindow('field2', cv2.WINDOW_NORMAL)
    # field2 = cv2.resize(field, (720,720), cv2.)
    # field2 = cv2.bitwise_not(field)
    cv2.imshow('field2', small_field)
    cv2.imshow('field', field)
    cv2.resizeWindow('field2', 720,720)
    while True:
        cv2.waitKey(1)
    