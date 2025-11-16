import numpy as np
from stl import Mesh
import vtkplotlib as vpl
import vtk


path1 = "HW5/case.stl"
mesh1 = Mesh.from_file(path1)

path2 = "HW5/mainshaft.stl"
mesh2 = Mesh.from_file(path2)
mainshaft_init = np.array([[-1, 0, 0, 500],
                           [0, 1, 0, 350],
                           [0, 0, -1, 340], 
                           [0, 0, 0, 1]])
# mainshaft_init = np.array([170,350,75])
# mesh2.rotate(np.array([0, 1, 0]), np.pi)
# mesh2.translate(mainshaft_init)
mesh2.transform(mainshaft_init)


path3 = "HW5/countershaft.stl"
mesh3 = Mesh.from_file(path3)
counter_init = np.array([0,100,70])
mesh3.translate(counter_init)

case_reader = vtk.vtkSTLReader()
case_reader.SetFileName('HW5/case.stl')
case_reader.Update()
case_mesh = case_reader.GetOutput()

main_reader = vtk.vtkSTLReader()
main_reader.SetFileName('HW5/mainshaft.stl')
main_reader.Update()
main_mesh = main_reader.GetOutput()

counter_reader = vtk.vtkSTLReader()
counter_reader.SetFileName('HW5/countershaft.stl')
counter_reader.Update()
counter_mesh = counter_reader.GetOutput()

t1 = vtk.vtkTransform()
t1.Identity()

t2 = vtk.vtkTransform()
# t2.RotateY(180.)
t2.SetMatrix(mainshaft_init.flatten())
# t2.Identity()

t3 = vtk.vtkTransform()
t3.Translate(counter_init)

print('starting')
case_collision = vtk.vtkCollisionDetectionFilter()
case_collision.SetInputData(0, case_mesh)
case_collision.SetTransform(0, t1)

case_collision.SetInputData(1, main_mesh)
case_collision.SetTransform(1, t2)

case_collision.SetBoxTolerance(0.0)      # spatial acceleration
case_collision.SetCellTolerance(0.0)
case_collision.SetNumberOfCellsPerNode(2)
case_collision.Update()

shaft_collision = vtk.vtkCollisionDetectionFilter()
shaft_collision.SetInputData(0, counter_mesh)
shaft_collision.SetTransform(0, t3)

shaft_collision.SetInputData(1, main_mesh)
shaft_collision.SetTransform(1, t2)

shaft_collision.SetBoxTolerance(0.0)      # spatial acceleration
shaft_collision.SetCellTolerance(0.0)
shaft_collision.SetNumberOfCellsPerNode(2)
shaft_collision.Update()

contacts = case_collision.GetNumberOfContacts() + shaft_collision.GetNumberOfContacts()

if contacts > 0:
    print("Meshes intersect!")
else:
    print("No intersection.")
    
    

vpl.mesh_plot(mesh1, color=np.array([255,255,255,125]))
vpl.mesh_plot(mesh2, color=(255,0,0))
vpl.mesh_plot(mesh3, color=(50,50,50))
vpl.show()