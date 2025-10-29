import sys
import cv2
import pygame
import numpy as np
import time
import math
from scipy.spatial import KDTree

from obstacle_field import FieldCreator
from firetruck import Firetruck
from road_map import RoadMap


MAP = 50
GRID_SIZE_M = 5
GRID_SIZE_PX = 15
PXS_PER_M = GRID_SIZE_PX//GRID_SIZE_M
FPS = 30

def display_map(field, map):
    for point in map.sample_poses:
        sample_box = cv2.RotatedRect((point[0],point[1]),(firetruck.length,firetruck.width),np.rad2deg(point[2]))
        pts = sample_box.points()*PXS_PER_M
        axle_loc = firetruck.get_axle_loc(point)
        cv2.drawContours(field.field,[pts.astype(np.int32).reshape((-1, 1, 2))],0,(255,0,0),1)
        cv2.arrowedLine(field.field,(round(axle_loc[0]*PXS_PER_M),round(axle_loc[1]*PXS_PER_M)),
                        (round(point[0]*PXS_PER_M), round(point[1]*PXS_PER_M)),(0,0,255),2)
        
    for edge in map.edge_map.values():
        path = edge["path"]
        for path_point in path:
            cv2.circle(field.field, (round(path_point[0]*PXS_PER_M), round(path_point[1]*PXS_PER_M)),1, (0,0,0),1)


if __name__ == '__main__':
    field = FieldCreator()
    field.createField(0.1, MAP, GRID_SIZE_PX)
    
    obstacles_pxs = np.where(cv2.cvtColor(field.field, cv2.COLOR_BGR2GRAY)<255)
    obstacles_m = np.vstack([obstacles_pxs[1], obstacles_pxs[0]]).T/PXS_PER_M
    kdtree = KDTree(obstacles_m)

    firetruck = Firetruck((127.5,247.5,np.deg2rad(-90)), kdtree)
    firetruck.draw(field.field, PXS_PER_M)
    # print(firetruck.check_collision())
    
    road_map = RoadMap(firetruck, (MAP-1)*GRID_SIZE_M, GRID_SIZE_M)
    road_map.build_road_map((127.5,247.5,np.deg2rad(-90)))
    display_map(field, road_map)
    
    cv2.imshow('field', field.field)
    while True:
        cv2.waitKey(1)
        