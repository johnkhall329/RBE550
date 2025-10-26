from queue import PriorityQueue
def path():
    pass

def valid():
    pass

def probability():
    pass

def reeds_shepp():
    pass

def heuristic():
    pass

def discretize(node, Resolution):
    return round(node/Resolution)

def find_neighbors(node):
    pass

def collision_check(node):
    pass

# Path Planner Pseudocode
def PathPlanner(Start, Goal, Map, Resolution, Ang_Resolution):
    frontier = PriorityQueue()
    came_from = {}
    cost_so_far = {}
    came_from[discretize(Start)] = None
    cost_so_far[discretize(Start)] = 0
    frontier.put((0,Start))
    
    while not frontier.empty():
        node = frontier.get()
        
        if collision_check(node): # check if current node is in collision
            continue
        
        if discretize(node) == discretize(Goal): # check if current node is in goal region
            return path(node)

        if probability(node): # calculate probability rs_path is generated from node based on distance-to-go/initial-distance
            rs_path = reeds_shepp(node, Goal)
            if not collision_check(rs_path): # check if Reeds Shepp path is viable
                return path(node) + rs_path
        
        for next_node in find_neighbors(node): # generate state lattice expansion with L, S, R, BL, B, BR directions
            if valid(node):
                new_cost = cost_so_far[discretize(node)] + next_node.additional_costs + Resolution # add additional costs of turning/backing up
                if next_node not in cost_so_far or new_cost < cost_so_far[discretize(next_node)]:
                    cost_so_far[discretize(next_node)] = new_cost
                    priority = new_cost + heuristic(next_node)
                    frontier.put(priority, next_node)
                    came_from[discretize(next_node)] = node