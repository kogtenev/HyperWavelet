import sys
import os
import numpy as np
import pymesh

def prepare_data(input_mesh, work_folder, duplicate_tol=1e-6):
    file = open(input_mesh, 'r')
    nobjects = int(file.readline())
    nmodules = int(file.readline())
    main_data = [nmodules, 0]
    current_size = 0
    for mod in range(nmodules):
        file.readline()
        ncells = int(file.readline())
        current_size += ncells
        main_data.append(current_size)
        file.readline()
        data = np.loadtxt(file, max_rows=ncells)
        points = data.reshape(-1, 3)
        cells = np.arange(points.shape[0], dtype=int)
        cells = cells.reshape(-1, 4)
        mesh = pymesh.form_mesh(points, cells)
        mesh, info = pymesh.remove_duplicated_vertices(mesh, duplicate_tol)
        pymesh.save_mesh(os.path.join(work_folder, 'mesh' + str(mod) + '.ply'), mesh, ascii=True)
        graph = pymesh.mesh_to_dual_graph(mesh)
        np.savetxt(os.path.join(work_folder, 'graph' + str(mod) + '.txt'), graph[1], fmt='%s')
    np.savetxt(os.path.join(work_folder, 'main.txt'), main_data, fmt='%s')
    
    
input_mesh = str(sys.argv[1])
work_folder = str(sys.argv[2])

work_folder = os.path.join(work_folder, 'input')

os.makedirs(work_folder, exist_ok=True)
prepare_data(input_mesh, work_folder)

