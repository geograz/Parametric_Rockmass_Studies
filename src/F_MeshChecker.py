# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:59:19 2024

@author: GEr
"""

import trimesh
from os import listdir

ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]

erroneous_meshes = ['243751266175', '769078262648', '926793301448']

for id_ in ids:
    if id_ in erroneous_meshes:
        pass
    else:
        fp = fr'../combinations/{id_}_discontinuities.stl'
        discontinuity_mesh = trimesh.load_mesh(fp)
        mesh_min, mesh_max = discontinuity_mesh.bounds
        small_side = min(mesh_max - mesh_min)
        large_side = max(mesh_max - mesh_min)
        print(id_, round(small_side, 3), round(large_side, 3))
        if small_side < 10:
            raise(ValueError(f'{id_} too small'))
        if large_side > 11:
            raise(ValueError(f'{id_} too large'))
