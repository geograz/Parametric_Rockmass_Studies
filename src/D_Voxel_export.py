# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:40:27 2024

@author: GEr
"""

import trimesh

RESOLUTIONS = [0.25, 0.2, 0.15, 0.1, 0.05]  # 3D grid resolution

sample = 151763271961

for resolution in RESOLUTIONS:
    fp = fr"..\combinations\{sample}_discontinuities.stl"
    discontinuity_mesh = trimesh.load_mesh(fp)
    discontinuity_voxels = discontinuity_mesh.voxelized(pitch=resolution)
    voxel_mesh = discontinuity_voxels.as_boxes()
    voxel_mesh.export(fr'..\output\{sample}_{resolution}_voxel_mesh.stl')
