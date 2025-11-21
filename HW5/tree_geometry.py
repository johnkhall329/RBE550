import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def convert_to_quat(x):
    """
    Convert x,y,z,r,p,y to x,y,z,qx,qy,qz,qw form
    """
    if len(x) == 6:
        r = Rotation.from_euler('xyz', x[3:], degrees=True)
        q = r.as_quat()
        x = normalize_quat(np.hstack([x[:3],q]))
    return x

def normalize_quat(x):
    """
    Ensure valid quaternions by normalizing and enforcing positive scalar
    """
    x[3:] /= np.linalg.norm(x[3:])
    if x[6] < 0:
        x[3:] *= -1
    return x

def weighted_dist(x, x_new, w_r=4.0):
    """
    Calculate distance as a function:
    dist = linear_dist + weight*rotation_dist
    This helps select nodes with added importance on orientation to reduce trapped positions.
    """
    d_p = np.linalg.norm(np.array(x_new[:3]) - np.array(x[:3]))
    r = Rotation.from_quat(x[3:])
    r_new = Rotation.from_quat(x_new[3:])
    r_rel = r * r_new.inv()
    return d_p + np.rad2deg(r_rel.magnitude())*w_r

def dist_between_points(a, b):
    """
    Return the Euclidean distance between two points
    :param a: first point
    :param b: second point
    :return: Euclidean distance between a and b
    """
    distance = np.linalg.norm(np.array(b) - np.array(a))
    return distance

def es_points_along_line(start, end, res_d, res_r):
    """
    Equally-spaced points along a line defined by start, end, with resolution
    :param start: starting point
    :param end: ending point
    :param res_d: maximum distance between points
    :param res_r: maximum rotational distance between points
    :return: yields points along line from start to end, separated by distance r
    """
    d= dist_between_points(start[:3], end[:3])
    r_start = Rotation.from_quat(start[3:])
    r_end = Rotation.from_quat(end[3:])
    r_rel = r_start.inv() * r_end
    dr = np.rad2deg(r_rel.magnitude())
    
    n_points = max(int(np.ceil(d / res_d)), int(np.ceil(dr / res_r)))
    n_points = max(n_points, 2)
    step_d = d / (n_points - 1)
    step_r = dr / (n_points - 1)
    for i in range(n_points):
        next_point = steer(start, end, i * step_d, i*step_r)
        yield next_point

def steer(start, goal, dp, dr):
    """
    Return a point in the direction of the goal, that is distance away from start
    :param start: start location
    :param goal: goal location
    :param dp: linear distance away from start
    :param dr: rotational distance away from start
    :return: point in the direction of the goal, distance away from start
    """
    start, goal = convert_to_quat(start), convert_to_quat(goal)
    v = goal[:3] - start[:3]
    u = v /np.linalg.norm(v)
    dp = min(dp, np.linalg.norm(v))
    new_pos = start[:3] + u*dp
    
    r_start = Rotation.from_quat(start[3:])
    r_goal = Rotation.from_quat(goal[3:])
    r_rel = r_start.inv() * r_goal
    angle = r_rel.magnitude()
    
    if angle > 1e-8:
        t = min(1.0, dr/np.rad2deg(angle))
        slerp = Slerp([0,1], Rotation.concatenate([r_start, r_goal])) # interpolate between quaternions using SLERP
        r_new = slerp(t)
    else:
        r_new = r_start
    
    steered_point = np.hstack([new_pos, r_new.as_quat()])
    return normalize_quat(steered_point)