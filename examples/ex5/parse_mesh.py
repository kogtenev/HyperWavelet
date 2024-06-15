import sys
import os
import numpy as np
import pymesh

def read_mesh(fname):
    file = open(fname, 'r')
    nobjects = int(file.readline())
    nmodules = int(file.readline())
    points = np.ndarray((0, 3))
    for mod in range(nmodules):
        file.readline()
        ncells = int(file.readline())
        file.readline()
        data = np.loadtxt(file, max_rows=ncells)
        data = data.reshape(-1, 3)
        points = np.vstack((points, data))
    cells = np.arange(points.shape[0], dtype=int)
    cells = cells.reshape(-1, 4)
    return pymesh.form_mesh(points, cells)
    
    
input_mesh = str(sys.argv[1])
output_folder = str(sys.argv[2])

os.mkdir(output_folder)

mesh = read_mesh(input_mesh)
mesh, info = pymesh.remove_duplicated_vertices(mesh, 1e-5)
pymesh.save_mesh(os.path.join(output_folder, 'mesh.ply'), mesh, ascii=True)

graph = pymesh.mesh_to_dual_graph(mesh)
np.savetxt(os.path.join(output_folder, 'graph.txt'), graph[1], fmt='%s')