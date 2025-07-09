# -*- coding: utf-8 -*-
"""
        PARAMETRIC ROCK MASS STUDIES
-- computational rock mass characterization --

Code author: Dr. Georg H. Erharter

Script that saves a mesh of a rastered sample. For visualization purposes only.
"""

import trimesh
from X_library import utilities

utils = utilities()

SAMPLE = 151763271961  # sample with ID to process
RESOLUTIONS = [0.25, 0.2, 0.15, 0.1, 0.05]  # 3D voxel resolution

for resolution in RESOLUTIONS:
    fp = utils.make_rel_fp(
        fr'sample_data\{SAMPLE}_discontinuities.stl', 1)
    discontinuity_mesh = trimesh.load_mesh(fp)
    discontinuity_voxels = discontinuity_mesh.voxelized(pitch=resolution)
    voxel_mesh = discontinuity_voxels.as_boxes()
    fp = utils.make_rel_fp(
        fr'sample_data\{SAMPLE}_{resolution}_voxel_mesh.stl', 1)
    voxel_mesh.export(fp)
