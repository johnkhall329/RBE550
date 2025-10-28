import sys
import cv2
import pygame
import numpy as np
import time
import math
from scipy.spatial import KDTree

from obstacle_field import FieldCreator
from firetruck import Firetruck

MAP = 50
GRID_SIZE_M = 5
GRID_SIZE_PX = 15
PXS_PER_M = GRID_SIZE_PX//GRID_SIZE_M
FPS = 30


if __name__ == '__main__':
    field = FieldCreator()
    field.createField(0.1, MAP, GRID_SIZE_PX)
    
    obstacles_pxs = np.where(cv2.cvtColor(field.field, cv2.COLOR_BGR2GRAY)<255)
    obstacles_m = np.vstack([obstacles_pxs[1], obstacles_pxs[0]]).T/PXS_PER_M
    kdtree = KDTree(obstacles_m)

    firetruck = Firetruck((125,125,np.deg2rad(20)))
    firetruck.draw(field.field, PXS_PER_M)
    print(firetruck.check_collision(kdtree))
    cv2.imshow('field', field.field)
    while True:
        cv2.waitKey(1)
        