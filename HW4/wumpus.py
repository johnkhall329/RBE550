from queue import PriorityQueue
import numpy as np
import cv2
import math

COLOR = (45,82,160)

class Wumpus():
    def __init__(self, start, obstacle_field, grid_m_conv):
        self.loc = start
        self.obstacle_field = obstacle_field.astype('int64')
        self.path = []
        self.conversion = grid_m_conv
        self.score = 0
        self.state = 'id'
        self.goal = None
        self.time = 0.0
    
    
    def draw(self, field, conversion):
        cv2.circle(field, (round(self.loc[0]*conversion), round(self.loc[1]*conversion)),3,COLOR,-1)

    def get_path(self, start_loc, goal_loc):
        frontier = PriorityQueue()
        frontier.put((0,start_loc)) # add start as first node with no cost and came from is None
        came_from = {}
        cost_so_far = {}
        came_from[start_loc] = None
        cost_so_far[start_loc] = 0
        goal_neighbors = find_neighbors(goal_loc)
        self.goal = goal_loc

        while not frontier.empty():
            item = frontier.get()
            curr_node = item[1]
            
            if curr_node in goal_neighbors: # if it arrives to a node adjacent to the goal return a path
                self.path = format_path(came_from, curr_node, self.conversion)
                return self.path
            
            # Iterates through closest 4 neighbors and see if they are within the field
            for next_node in find_neighbors(curr_node):
                if 0<=next_node[0]<self.obstacle_field.shape[0] and 0<=next_node[1]<self.obstacle_field.shape[1] and self.obstacle_field[next_node] != 255:
                    new_cost = cost_so_far[curr_node] + self.obstacle_field[next_node] + 1
                    prev_cost = cost_so_far.get(next_node)
                    if prev_cost is None or new_cost < prev_cost: # only adds to queue if unvisited or cheaper to get to
                        cost_so_far[next_node] = new_cost
                        heuristic = math.sqrt((goal_loc[0]-next_node[0])**2 + (goal_loc[1]-next_node[1])**2) # Euclidean distance to goal
                        priority = int(new_cost + heuristic)
                        frontier.put((priority,next_node))
                        came_from[next_node] = curr_node
        
        raise ValueError("No Path Found")
    
def find_neighbors(curr_node):
    neighbors = [(curr_node[0]-1, curr_node[1]),
                (curr_node[0], curr_node[1]+1),
                (curr_node[0]+1, curr_node[1]),
                (curr_node[0], curr_node[1]-1)]
    return neighbors
    
def format_path(came_from, goal_loc, conversion):
    path = []
    node = goal_loc
    while node is not None: # appends nodes in path with goal as beginning
        path.insert(0,((node[1]+0.5)*conversion, (node[0]+0.5)*conversion))
        node = came_from[node]
    return path