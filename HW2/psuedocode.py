from queue import PriorityQueue
start_loc = 1
goal=1

def formated_path():
    pass

def valid():
    pass

def heuristic():
    pass

# Path Planner:
def Path_Planner(goal, start_loc):
    frontier = PriorityQueue()
    frontier.put((0,start_loc))
    came_from = {}
    cost_so_far = {}
    came_from[start_loc] = None
    cost_so_far[start_loc] = 0

    while not frontier.empty():
        current_node = frontier.get()
        if current_node == goal:
            return formated_path()
        
        for next_node in current_node.neighbors(): # +/- (1,1)
            if valid(next_node): # Within field and not a wall
                new_cost = cost_so_far[current_node] + next_node.cost()
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(next_node)
                    frontier.put((priority,next_node))
                    came_from[next_node] = current_node