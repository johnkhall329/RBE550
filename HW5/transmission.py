import vtk 
from stl import Mesh
import vtkplotlib as vpl
import numpy as np
import time
import pickle
from scipy.spatial.transform import Rotation

from search_space import CollisionChecker
from birrt import BiRRT

def init_display(main_path, counter_path, case_path, starting_pos):
    """
    Display initial setup of transmission
    """
    case_mesh = Mesh.from_file(case_path)

    main_mesh = Mesh.from_file(main_path)
    m_origin = np.array([330,130,130])
    main_mesh.translate(-m_origin)
    main_mesh.rotate(np.array([0,1,0]), np.pi)
    main_mesh.translate(starting_pos)

    counter_mesh = Mesh.from_file(counter_path)
    counter_origin = np.array([330,140,140])
    counter_init = np.array([330,250,210])
    counter_mesh.translate(-counter_origin)
    counter_mesh.translate(counter_init)

    fig = vpl.figure("Transmission")
    vpl.mesh_plot(case_mesh, color=np.array([255,255,255,50]), fig=fig)
    vpl.mesh_plot(main_mesh, color=(255,0,0,100), fig=fig)
    vpl.mesh_plot(counter_mesh, color=(50,50,50,100), fig=fig)
    vpl.scatter(np.array([[0,0,0]]),color=(0,0,0),radius=5, fig=fig)
    fig.show(block=False)
    return fig, [case_mesh, main_mesh, counter_mesh]

def load_from_file():
    """
    Load path from file
    """
    path = np.load('HW5/path.npy')
    with open("HW5/tree0.pkl", "rb") as f:
        t0_e = pickle.load(f)
    with open("HW5/tree1.pkl", "rb") as f2:
        t1_e = pickle.load(f2)
    return path, t0_e, t1_e

def display_path(tree_edges, start, goal, fig):
    """
    Display BiRRT search tree in 3D
    """
    start_pos = np.array(start)
    goal_pos = np.array(goal)
    vpl.scatter([start_pos],[0,0,0],radius=2, fig=fig)
    vpl.scatter([goal_pos],[0,0,0],radius=2, fig=fig)
    for i,edges in enumerate(tree_edges):
        for p1, p2 in edges.items():
            if p2 is None: continue
            p1 = np.array(p1[:3]) + start_pos
            p2 = np.array(p2[:3]) + start_pos
            vpl.scatter([p1], radius=3,color=(0,0,255),fig=fig)
    fig.update()
    
def animate_path(path, starting_pos, meshes):
    """
    Take a successful path and starting position and animate the transmission
    """
    path_fig = vpl.figure("Path Animation")
    main_mesh = meshes[1]
    
    vpl.mesh_plot(meshes[0], color=np.array([255,255,255,100]), fig=path_fig)
    vpl.mesh_plot(meshes[2], color=(255,255,255,100), fig=path_fig)
    prev_pos = np.eye(4)
    prev_pos[:3,3] = starting_pos
    prev_mesh = vpl.mesh_plot(main_mesh, color=(255,0,0), fig=path_fig)
    path_fig.show(block=False)
    for pose in path:
        R = Rotation.from_quat(pose[3:])
        t = pose[:3]
        T = np.eye(4)
        T[:3,3] = t+starting_pos
        T[:3,:3] = R.as_matrix()     
        
        path_fig.remove_plot(prev_mesh)
        main_mesh.transform(np.linalg.inv(prev_pos))
        main_mesh.transform(T)
        vpl.scatter([T[:3,3]], radius=5, color=(0,0,0), fig = path_fig)
        prev_pos = T
        prev_mesh = vpl.mesh_plot(main_mesh, color=(255,0,0), fig=path_fig)
        path_fig.update()
        path_fig.show(block=False)
        time.sleep(0.05)
    path_fig.show()

if __name__ == '__main__':
    case_path = "HW5/case.stl"
    main_path = "HW5/mainshaft.stl"
    counter_path = "HW5/countershaft.stl"
    
    x_init =  (0,0,0,0,0,0)
    x_goal = (250,400,0,0,0,0)
    starting_pos = np.array([170, 480, 210])
    fig, meshes = init_display(main_path, counter_path, case_path, starting_pos)
    cc = CollisionChecker(main_path,counter_path,case_path)
    path = None
    while path is None:
        path_planner = BiRRT(cc, 5, 5, x_init, x_goal, 5e4, 1, 0.5, 750)
        path, t0_e, t1_e = path_planner.rrt_connect()
    # path, t0_e, t1_e = load_from_file()
    path_goal = x_goal[:3] + starting_pos
    display_path([t0_e, t1_e], starting_pos, path_goal, fig)
    
    fig.show(block=False)
    animate_path(path, starting_pos, meshes)
