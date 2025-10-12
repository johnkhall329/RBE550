import cv2
import numpy as np
import math

CAR_COLOR = (255,0,0)
PATH_COLOR = (255,150,0)

class Vehicle():
    def __init__(self, type, size, scale, origin):
        self.type = type
        self.scale = scale
        self.img_size = size
        if self.type == 'delivery':
            self.width = 0.57*scale
            self.height = 0.7*scale
            self.x = origin[0]
            self.y = origin[1]
        elif self.type == 'car':
            self.width = 1.8*scale
            self.height = 5.2*scale
            self.wheelbase = 2.8
            self.x = origin[0]
            self.y = origin[1] - self.wheelbase/2
        elif self.type == 'truck':
            self.width = 2.0*scale
            self.height = 5.4*scale
            self.wheelbase = 3.4
            self.trailer_width = 2.0*scale
            self.trailer_height = 4.5*scale
            self.trailer_dist = 5.0*scale
            self.x = origin[0]
            self.y = origin - self.wheelbase/2 + self.trailer_height
            self.trailer_heading = math.pi/2
            self.trailer_center = (self.x-(self.trailer_dist*math.cos(self.trailer_heading)), self.y-(self.trailer_dist*math.sin(self.trailer_heading)))
            self.trailer_points = []         
        self.box_points = []
        self.path_box_points = []
        self.heading = math.pi/2 # point along y axis
        self.speed = 0.0
        for i in [-1,1]:
            for j in [-1,1]:
                self.box_points.append((int(origin[0] + i*self.width/2), int(origin[1] + j*self.height/2)))
                if self.type == 'truck':
                    self.trailer_points.append((int(self.trailer_center[0] + i*self.trailer_width/2), int(self.trailer_center[1] + j*self.trailer_height/2)))
        self.box_points = np.array(self.box_points)
        self.path_box_points.append(self.box_points)
        if self.type == 'truck': 
            self.trailer_points = np.array(self.trailer_points)
            self.path_box_points.append(self.trailer_points)
                    
    def update(self, U, dt):
        theta, a = U
        v_dot = a  # acceleration
        self.speed += v_dot * dt
        phi_dot = self.speed * math.tan(theta)/self.wheelbase*dt if self.type != 'delivery' else phi_dot = theta * dt # simple bicycle model
        self.heading += phi_dot
        dx = self.speed * math.sin(self.heading) * dt
        dy = self.speed * math.cos(self.heading) *dt
        self.x += dx
        self.y += dy
        
        for i,point in enumerate(self.box_points):
            point = (point[0]+dx, point[1]+dy)
            new_x = int((point[0]-self.x)*math.cos(phi_dot) - (point[1]-self.y)*math.sin(phi_dot) + self.x)
            new_y = int((point[1]-self.y)*math.cos(phi_dot) + (point[0]-self.x)*math.sin(phi_dot) + self.y)
            self.box_points[i] = (new_x,new_y)
        self.path_box_points.append(self.box_points)
                
        if self.type == 'truck':
            dt_trailer = self.speed/self.trailer_dist*math.sin(self.trailer_heading,self.heading)
            self.trailer_heading += dt_trailer
            new_trailer_center = (self.x-(self.trailer_dist*math.cos(self.trailer_heading)), self.y-(self.trailer_dist*math.sin(self.trailer_heading)))
            dx_trailer = new_trailer_center[0] - self.trailer_center[0]
            dy_trailer =  new_trailer_center[1] - self.trailer_center[1]
            self.trailer_center = new_trailer_center
            for i,point in enumerate(self.trailer_points):
                point = (point[0]+dx_trailer, point[1]+dy_trailer)
                new_x = int((point[0]-self.trailer_center[0])*math.cos(dt_trailer) - (point[1]-self.y)*math.sin(dt_trailer) + self.trailer_center[0])
                new_y = int((point[1]-self.y)*math.cos(dt_trailer) + (point[0]-self.x)*math.sin(dt_trailer) + self.trailer_center[1])
                self.trailer_points[i] = (new_x,new_y)
            self.path_box_points.append(self.trailer_points)
        
            
    def draw(self,color_field, bin_field, path=True):
        cv2.fillConvexPoly(color_field,self.box_points,CAR_COLOR,-1)
        cv2.fillConvexPoly(bin_field,self.box_points, (255,255,255), -1)
        
        if path:
            for box in self.path_box_points:
                cv2.drawContours(color_field,[box],0,PATH_COLOR,2)
    