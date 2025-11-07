import cv2
import numpy as np
import math
from scipy.spatial import KDTree

COLOR = (0,0,255)

class Firetruck():
    def __init__(self, start, tree:KDTree, conversion):
        self.x, self.y, self.heading = start
        self.length = 4.9
        self.width = 2.2
        self.wheelbase = 3.0
        self.turning_r = 13.0
        self.v = 10.0
        self.collision_r = math.ceil(math.sqrt((self.length/2)**2 + (self.width/2)**2)) # collision radius is from center to corner rounded up
        self.box = cv2.RotatedRect((self.x,self.y),(self.length,self.width),np.rad2deg(self.heading))
        self.tree = tree
        self.state = 'id'
        self.conversion = conversion
        self.goal = None
        self.extinguishing = {}
        self.score = 0
        self.time = 0.0
        
    def check_collision(self, pose=None, axle = False):
        '''
        Check for collisions for the current location or a given pose based on the center or axle location
        Returns true if there is a collision
        '''
        x,y,heading = [self.x, self.y, self.heading] if pose is None else pose
        if axle:
            x,y,heading = self.get_center((x,y,heading))
        collided_idxs = self.tree.query_ball_point((x,y),self.collision_r) # get obstacle points within collision radius
        if len(collided_idxs) > 0:
            collision_box = cv2.RotatedRect((x,y),(self.length, self.width),np.rad2deg(heading))
            for obstacle in self.tree.data[collided_idxs]:
                if cv2.pointPolygonTest(collision_box.points(), (obstacle[0], obstacle[1]), False) >= 0: return True # use polygon test to determine if obstacle point lays within firetruck rectangle
        return False
    
    def draw(self, field):
        self.box = cv2.RotatedRect((self.x,self.y),(self.length,self.width),np.rad2deg(self.heading))
        pts = self.box.points()*self.conversion
        axle_loc = self.get_axle_loc()
        cv2.fillConvexPoly(field,pts.astype(np.int32).reshape((-1, 1, 2)),COLOR)
        cv2.arrowedLine(field,(round(axle_loc[0]*self.conversion),round(axle_loc[1]*self.conversion)),(round(self.x*self.conversion), round(self.y*self.conversion)),(255,0,0),2)
        
    def get_axle_loc(self, pose=None):
        """
        Get the location of the firetruck's axle
        """
        x,y,heading = [self.x, self.y, self.heading] if pose is None else pose
        axle_x = x-(math.cos(heading)*self.wheelbase/2)
        axle_y = y-(math.sin(heading)*self.wheelbase/2)
        return (axle_x, axle_y, heading)
        
    def get_center(self, pose):
        """
        Get the location of the firetruck center
        """
        x,y,heading = pose
        center_x = x+(math.cos(heading)*self.wheelbase/2)
        center_y = y+(math.sin(heading)*self.wheelbase/2)
        return (center_x, center_y, heading)