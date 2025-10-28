import cv2
import numpy as np
import math
from scipy.spatial import KDTree

COLOR = (0,0,255)

class Firetruck():
    def __init__(self, start):
        self.x, self.y, self.heading = start
        self.length = 4.9
        self.width = 2.2
        self.wheelbase = 3.0
        self.turning_r = 13.0
        self.v = 10.0
        self.collision_r = math.sqrt((self.length/2)**2 + (self.width)**2)
        self.box = cv2.RotatedRect((self.x,self.y),(self.length,self.width),np.rad2deg(self.heading))
        
    def check_collision(self, tree:KDTree, pose=None):
        x,y,heading = [self.x, self.y, self.heading] if pose is None else pose
        collided_idxs = tree.query_ball_point((x,y),self.collision_r)
        if len(collided_idxs) > 0:
            collision_box = cv2.RotatedRect((x,y),(self.length, self.width),np.rad2deg(heading))
            for obstacle in tree.data[collided_idxs]:
                if cv2.pointPolygonTest(collision_box.points(), (obstacle[0], obstacle[1]), False) >= 0: return False
        return True
    
    def draw(self, field, conversion):
        self.box = cv2.RotatedRect((self.x,self.y),(self.length,self.width),np.rad2deg(self.heading))
        pts = self.box.points()*conversion
        axle_loc = self.get_axle_loc()
        cv2.fillConvexPoly(field,pts.astype(np.int32).reshape((-1, 1, 2)),COLOR)
        cv2.arrowedLine(field,(round(axle_loc[0]*conversion),round(axle_loc[1]*conversion)),(round(self.x*conversion), round(self.y*conversion)),(255,0,0),2)
        
    def get_axle_loc(self, pose=None):
        x,y,heading = [self.x, self.y, self.heading] if pose is None else pose
        axle_x = x-(math.cos(heading)*self.wheelbase/2)
        axle_y = y-(math.sin(heading)*self.wheelbase/2)
        return (axle_x, axle_y)
        