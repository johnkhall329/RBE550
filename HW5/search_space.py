import numpy as np
from stl import Mesh
import vtkplotlib as vpl
import vtk
from scipy.spatial.transform import Rotation

class CollisionChecker():
    def __init__(self, main_path, counter_path, case_path):
        """
        Load STLs and initialize vtk collision detection filters for countershaft and case
        """
        case_reader = vtk.vtkSTLReader()
        case_reader.SetFileName(case_path)
        case_reader.Update()
        self.case_mesh = case_reader.GetOutput()

        main_reader = vtk.vtkSTLReader()
        main_reader.SetFileName(main_path)
        main_reader.Update()
        self.main_mesh = main_reader.GetOutput()

        counter_reader = vtk.vtkSTLReader()
        counter_reader.SetFileName(counter_path)
        counter_reader.Update()
        self.counter_mesh = counter_reader.GetOutput()
        
        self.m_origin = np.array([330,130,130])
        self.mainshaft_init = np.array([[1, 0, 0, 170],
                                        [0, 1, 0, 480],
                                        [0, 0, 1, 210], 
                                        [0, 0, 0, 1]])
        
        self.counter_origin = np.array([330,140,140])
        self.counter_init = np.array([330,250,210])
        
        self.case_t = vtk.vtkTransform()
        self.case_t.Identity()

        self.main_t = vtk.vtkTransform()
        self.main_t.Translate(-self.m_origin[0], -self.m_origin[1], -self.m_origin[2])
        self.main_t.SetMatrix(self.mainshaft_init.flatten())

        self.counter_t = vtk.vtkTransform()
        self.counter_t.Translate(self.counter_init)
        self.counter_t.Translate(-self.counter_origin)

        self.case_collision = vtk.vtkCollisionDetectionFilter()
        self.case_collision.SetInputData(0, self.case_mesh)
        self.case_collision.SetTransform(0, self.case_t)

        self.case_collision.SetInputData(1, self.main_mesh)
        self.case_collision.SetTransform(1, self.main_t)

        self.case_collision.SetBoxTolerance(0.0)      # spatial acceleration
        self.case_collision.SetCellTolerance(0.0)
        self.case_collision.SetNumberOfCellsPerNode(2)
        self.case_collision.Update()

        self.shaft_collision = vtk.vtkCollisionDetectionFilter()
        self.shaft_collision.SetInputData(0, self.counter_mesh)
        self.shaft_collision.SetTransform(0, self.counter_t)

        self.shaft_collision.SetInputData(1, self.main_mesh)
        self.shaft_collision.SetTransform(1, self.main_t)

        self.shaft_collision.SetBoxTolerance(0.0)      # spatial acceleration
        self.shaft_collision.SetCellTolerance(0.0)
        self.shaft_collision.SetNumberOfCellsPerNode(2)
        self.shaft_collision.Update()

    def collision_free(self, x, deg = True):
        """
        Given a mainshaft pose, transform the stl to the pose and check for collisions
        Returns True if there are no collisions detected
        """
        main_t = np.eye(4)
        main_t[:3, 3] = x[:3]
        if len(x) == 6:
            R = Rotation.from_euler('xyz', x[3:],degrees = deg)
        else:
            R = Rotation.from_quat(x[3:])
        main_t[:3,:3]=R.as_matrix()
        new_T = vtk.vtkTransform()
        # new_T.Translate(m_origin[0], m_origin[1], m_origin[2])
        T = self.mainshaft_init@main_t
        new_T.SetMatrix(T.flatten())
        new_T.RotateY(180)
        new_T.Translate(-self.m_origin[0], -self.m_origin[1], -self.m_origin[2])
        
        self.case_collision.SetTransform(1, new_T)
        self.shaft_collision.SetTransform(1, new_T)
        self.case_collision.Update()
        self.shaft_collision.Update()
        
        contacts = self.case_collision.GetNumberOfContacts() + self.shaft_collision.GetNumberOfContacts()
        return contacts <= 0
    
    def sample_free(self, bound):
        """
        Sample within bounds until there is a pose that is collision free
        """
        x,y,z = np.random.randint(-bound, bound, 3)
        r,p,y = np.random.randint(-179, 180, 3)
        if self.collision_free(np.array([x,y,z,r,p,y])):
            return np.array([x,y,z,r,p,y])
        else: return self.sample_free(bound)
        
            
if __name__ == '__main__':
    path1 = "HW5/case.stl"
    mesh1 = Mesh.from_file(path1)

    path2 = "HW5/mainshaft.stl"
    mesh2 = Mesh.from_file(path2)
    start_pos = np.array([170, 480, 210, 0,0,0])
    change = np.array([250, 400, 0, 0, 0, 0])
    new_pos = start_pos + change
    # mainshaft_init = np.array([170,350,75])
    # mesh2.rotate(np.array([0, 1, 0]), np.pi)
    # mesh2.translate(mainshaft_init)
    main_t = np.eye(4)
    main_t[:3, 3] = new_pos[:3]
    R = Rotation.from_euler('xyz', new_pos[3:])
    main_t[:3,:3]=R.as_matrix()

    m_origin = np.array([330,130,130])
    mesh2.translate(-m_origin)
    mesh2.rotate(np.array([0,1,0]), np.pi)
    mesh2.transform(main_t)

    path3 = "HW5/countershaft.stl"
    mesh3 = Mesh.from_file(path3)
    counter_origin = np.array([330,140,140])
    counter_init = np.array([330,250,210])
    mesh3.translate(-counter_origin)
    mesh3.translate(counter_init)

    cc = CollisionChecker(path2, path3, path1)
    print(cc.collision_free(change))

    fig = vpl.figure("Transmission")
    vpl.mesh_plot(mesh1, color=np.array([255,255,255,125]), fig=fig)
    vpl.mesh_plot(mesh2, color=(255,0,0,125), fig=fig)
    vpl.mesh_plot(mesh3, color=(50,50,50,125), fig=fig)
    vpl.scatter(np.array([[0,0,0]]),color=(0,0,0),radius=5, fig=fig)
    fig.update()
    fig.show()
    # vpl.show()