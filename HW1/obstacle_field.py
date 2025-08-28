import numpy as np
import cv2

def createField(coverage, map_size, cell_size):
    field = np.ones((map_size*cell_size,map_size*cell_size))
    cv2.namedWindow(f'{100*coverage}% coverage', cv2.WINDOW_NORMAL)
    cv2.imshow(f'{100*coverage}% coverage', field)
    # cv2.resizeWindow(f'{100*coverage}% coverage', 720, 720)
    while cv2.countNonZero(field)/(map_size*cell_size)**2 > (1.0-coverage):
        shape_type = np.random.uniform(0, 4)
        loc = (int(np.random.uniform(0,map_size-2)),int(np.random.uniform(0,map_size)))
        
        if (shape_type > 3):
            field = addI(field, loc, map_size, cell_size)
        elif (shape_type > 2):
            field = addL(field, loc, map_size, cell_size)
        elif (shape_type > 1):
            field = addS(field, loc, map_size, cell_size)
        else:
            field = addT(field, loc, map_size, cell_size)
        # cv2.imshow(f'{100*coverage}% coverage', field)
        # cv2.waitKey(1)
    cv2.imshow(f'{100*coverage}% coverage', field)
    return field
    
    
def addI(field, loc, map_size, cell_size):
    cv2.rectangle(field, (cell_size*loc[1], cell_size*loc[0]), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+1)-1), 0, thickness=cv2.FILLED)
    # cv2.rectangle(field, loc, (loc[0]+1, loc[1]+1), 0, thickness=cv2.FILLED)
    if loc[0] + 1 < map_size:
        cv2.rectangle(field, (cell_size*loc[1], cell_size*(loc[0]+1)), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+2)-1), 0, thickness=cv2.FILLED) 
    if loc[0] + 2 < map_size: 
        cv2.rectangle(field, (cell_size*loc[1], cell_size*(loc[0]+2)), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+3)-1), 0, thickness=cv2.FILLED) 
    if loc[0] + 3 < map_size: 
        cv2.rectangle(field, (cell_size*loc[1], cell_size*(loc[0]+3)), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+4)-1), 0, thickness=cv2.FILLED) 
    return field

def addL(field, loc, map_size, cell_size):
    cv2.rectangle(field, (cell_size*loc[1], cell_size*loc[0]), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+1)-1), 0, thickness=cv2.FILLED)
    if loc[0] + 1 < map_size: 
        cv2.rectangle(field, (cell_size*loc[1], cell_size*(loc[0]+1)), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+2)-1), 0, thickness=cv2.FILLED)
    if loc[0] + 2 < map_size: 
        cv2.rectangle(field, (cell_size*loc[1], cell_size*(loc[0]+2)), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+3)-1), 0, thickness=cv2.FILLED)
    if loc[0] + 3 < map_size and loc[1] + 1 < map_size: 
        cv2.rectangle(field, (cell_size*(loc[1]+1), cell_size*(loc[0]+2)), (cell_size*(loc[1]+2)-1, cell_size*(loc[0]+3)-1), 0, thickness=cv2.FILLED)
    return field

def addS(field, loc, map_size, cell_size):
    cv2.rectangle(field, (cell_size*loc[1], cell_size*loc[0]), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+1)-1), 0, thickness=cv2.FILLED)
    if loc[0] + 1 < map_size: 
        cv2.rectangle(field, (cell_size*loc[1], cell_size*(loc[0]+1)), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+2)-1), 0, thickness=cv2.FILLED)
    if loc[0] + 1 < map_size and loc[1] + 1 < map_size: 
        cv2.rectangle(field, (cell_size*(loc[1]+1), cell_size*(loc[0]+1)), (cell_size*(loc[1]+2)-1, cell_size*(loc[0]+2)-1), 0, thickness=cv2.FILLED)
    if loc[0] + 2 < map_size and loc[1] + 1 < map_size: 
        cv2.rectangle(field, (cell_size*(loc[1]+1), cell_size*(loc[0]+2)), (cell_size*(loc[1]+2)-1, cell_size*(loc[0]+3)-1), 0, thickness=cv2.FILLED)
    return field

def addT(field, loc, map_size, cell_size):
    cv2.rectangle(field, (cell_size*loc[1], cell_size*loc[0]), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+1)-1), 0, thickness=cv2.FILLED)
    if loc[0] + 1 < map_size: 
        cv2.rectangle(field, (cell_size*loc[1], cell_size*(loc[0]+1)), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+2)-1), 0, thickness=cv2.FILLED)
    if loc[0] + 1 < map_size and loc[1] + 1 < map_size: 
        cv2.rectangle(field, (cell_size*(loc[1]+1), cell_size*(loc[0]+1)), (cell_size*(loc[1]+2)-1, cell_size*(loc[0]+2)-1), 0, thickness=cv2.FILLED)
    if loc[0] + 2 < map_size: 
        cv2.rectangle(field, (cell_size*loc[1], cell_size*(loc[0]+2)), (cell_size*(loc[1]+1)-1, cell_size*(loc[0]+3)-1), 0, thickness=cv2.FILLED)
    return field

if __name__ == '__main__':
    ten = createField(0.1, 128, 10)
    fifty =  createField(0.5, 128, 3)
    seventy = createField(0.7, 128, 1)
    twenty = createField(0.2, 64, 10)
    while True:
        cv2.waitKey(1)
    