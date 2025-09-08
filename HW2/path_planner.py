from queue import PriorityQueue
import cv2
import numpy as np

def get_path(field, goal_loc, start_loc):
    frontier = PriorityQueue()
    frontier.put((0,start_loc))
    came_from = {}
    cost_so_far = {}
    came_from[start_loc] = None
    cost_so_far[start_loc] = 0
    colored_field = cv2.cvtColor(field,cv2.COLOR_GRAY2BGR)
    field = field.astype('int64')
    state = np.zeros_like(field,dtype=np.uint8)
    max_state = 0
    cv2.namedWindow('state', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('state', 640,640)
    print(frontier.queue)
    while not frontier.empty():
        item = frontier.get()
        curr_node = item[1]
        
        if curr_node == goal_loc:
            return format_path(came_from, curr_node)
        
        for next_node in [(curr_node[0]-1, curr_node[1]),(curr_node[0], curr_node[1]+1),(curr_node[0]+1, curr_node[1]),(curr_node[0], curr_node[1]-1)]:
            if 0<=next_node[0]<field.shape[0] and 0<=next_node[1]<field.shape[1] and field[next_node] != 255:
                new_cost = cost_so_far[curr_node] + field[next_node] + 1
                prev_cost = cost_so_far.get(next_node)
                if prev_cost is None or new_cost < prev_cost:
                    cost_so_far[next_node] = new_cost
                    heuristic = abs(goal_loc[0]-next_node[0]) + abs(goal_loc[1]-next_node[1])
                    priority = int(new_cost + heuristic)
                    frontier.put((priority,next_node))
                    state[next_node]=new_cost
                    max_state = max(max_state,new_cost)
                    came_from[next_node] = curr_node
        
        state2 = (state/max_state*255).astype('uint8')
        colored_state = cv2.applyColorMap(state2, cv2.COLORMAP_JET)
        path_state = cv2.addWeighted(colored_state,0.5, colored_field,0.5,1.)
        path_state[goal_loc] = [0,255,0]
        path_state[start_loc] = [255,255,255]
        cv2.imshow('state', path_state)
        cv2.waitKey(10)

    raise ValueError("No Path Found")

def format_path(came_from, goal_loc):
    path = []
    node = goal_loc
    while came_from[node] is not None:
        path.append(node)
        node = came_from[node]
    return path