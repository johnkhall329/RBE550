import cv2
import numpy as np
import math

CAR_COLOR = (255,0,0)
PATH_COLOR = (150,150,0)

class Vehicle():
    def __init__(self, type, img_size, origin):
        self.type = type
        self.img_size = img_size
        scale = self.img_size/12/3
        if self.type == 'delivery':
            self.width = 0.57*scale
            self.height = 0.7*scale
            self.x = origin[0]
            self.y = origin[1]
            self.wheelbase = 0
        elif self.type == 'car':
            self.width = 1.8*scale
            self.height = 5.2*scale
            self.wheelbase = 2.8*scale
            self.x = origin[0]
            self.y = origin[1] - self.wheelbase/2
        elif self.type == 'truck':
            self.width = 2.0*scale
            self.height = 5.4*scale
            self.wheelbase = 3.4*scale
            self.trailer_width = 2.0*scale
            self.trailer_height = 4.5*scale
            self.trailer_dist = 5.0*scale
            self.x = origin[0]
            self.y = origin[1] - self.wheelbase/2
            self.trailer_heading = math.pi/2
            self.trailer_center = (self.x-(self.trailer_dist*math.cos(self.trailer_heading)), self.y-(self.trailer_dist*math.sin(self.trailer_heading)))     
        self.path_box_points = []
        self.heading = math.pi/2 # point along y axis
        self.speed = 0.0
        self.box = cv2.RotatedRect((origin[0],origin[1]),(self.height,self.width),np.rad2deg(self.heading))
        self.path_box_points.append(np.int32(self.box.points()))
        if self.type == 'truck': 
            self.trailer_box = cv2.RotatedRect((self.trailer_center[0],self.trailer_center[1]),((self.trailer_height),(self.trailer_width)),np.rad2deg(self.trailer_heading))
            self.path_box_points.append(np.int32(self.trailer_box.points()))
                    
    def update(self, U, dt):
        theta, v = U
        self.speed = v
        phi_dot = self.speed * math.tan(theta)/self.wheelbase*dt if self.type != 'delivery' else theta * dt # simple bicycle model
        self.heading += phi_dot
        dx = self.speed * math.cos(self.heading) * dt
        dy = self.speed * math.sin(self.heading) *dt
        self.x += dx
        self.y += dy
        
       
        # print(self.box_points)
        center = (self.x+(math.cos(self.heading)*self.wheelbase/2),self.y+(math.sin(self.heading)*self.wheelbase/2))
        self.box.center = center
        self.box.angle = np.rad2deg(self.heading)
        self.path_box_points.append(np.int32(self.box.points()))
            
        if self.type == 'truck':
            self.trailer_heading -= self.speed/self.trailer_dist*math.sin(self.trailer_heading-self.heading)*dt
            self.trailer_center = (self.x-(self.trailer_dist*math.cos(self.trailer_heading)), self.y-(self.trailer_dist*math.sin(self.trailer_heading)))
            self.trailer_box.center = self.trailer_center
            self.trailer_box.angle = np.rad2deg(self.trailer_heading)
            self.path_box_points.append(np.int32(self.trailer_box.points()))
        
            
    def draw(self,color_field_copy, color_field, path=False):
        pts = self.box.points().astype(np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(color_field_copy,pts,CAR_COLOR)
        center = (round(self.x+(math.cos(self.heading)*10)),round(self.y+(math.sin(self.heading)*10)))
        cv2.arrowedLine(color_field_copy,(round(self.x),round(self.y)),center,(0,0,255),2)
        if self.type == 'truck': 
            t_pts = self.trailer_box.points().astype(np.int32).reshape((-1, 1, 2))
            cv2.fillConvexPoly(color_field_copy,t_pts,CAR_COLOR)
            cv2.line(color_field_copy,(round(self.x),round(self.y)),(round(self.trailer_center[0]),round(self.trailer_center[1])),(0,0,0),2)
        
        if path:
            cv2.drawContours(color_field,[pts],0,PATH_COLOR,1)
            if self.type == 'truck': cv2.drawContours(color_field,[t_pts],0,PATH_COLOR,1)
        
        return color_field_copy, color_field