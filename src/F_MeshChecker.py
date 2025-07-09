# -*- coding: utf-8 -*-
"""
        PARAMETRIC ROCK MASS STUDIES
-- computational rock mass characterization --

Code author: Dr. Georg H. Erharter

Script that performs different checks of the generated meshes.
"""

from os import listdir

import trimesh

# get ids of all generated meshes
ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]

# check if meshes are too large or too small (must be within 10x10x10 cube)
erroneous_meshes = []

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
        if large_side > 10.2:
            raise(ValueError(f'{id_} too large'))
