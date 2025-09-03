from queue import PriorityQueue

def get_path(field, goal_loc, start_loc):
    frontier = PriorityQueue()
    frontier.put((0,start_loc))
    came_from = {}
    cost_so_far = {}
    came_from[start_loc] = None
    cost_so_far[start_loc] = 0
    field = field.astype('int64')

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
                    came_from[next_node] = curr_node
    
    raise ValueError("No Path Found")

def format_path(came_from, goal_loc):
    path = []
    node = goal_loc
    while came_from[node] is not None:
        path.append(node)
        node = came_from[node]
    return path