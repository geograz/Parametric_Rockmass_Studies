# -*- coding: utf-8 -*-
"""
Experimental code to the paper XXXX
Dr. Georg H. Erharter - 2023
DOI: https://doi.org/10.1007/s00603-024-03787-9

Script that performs different raster based analyses on the dataset
(results not included in first paper).
"""

import gc
import numpy as np
import open3d as o3d
from os import listdir

from X_library import parameters, utilities

#############################
# static variables and constants

N_SETS_TO_PROCESS = 47  # max number of sets to process in this run

TOT_BBOX_SIZE = 10  # total bounding box size [m]
UNIT_BOX_SIZE = 1  # size of a measurement box [m]
RESOLUTION = 256  # 3D grid resolution, 256
voxel_size = TOT_BBOX_SIZE / RESOLUTION  # .05
color = 1
DICE_THRESH = 0.75  # threshold of dice coefficient that indicates similarity
# run code for random- or sequential unprocessed samples -> multiprocessing
MODE = 'random'  # 'random', 'sequential'
COMPUTE_SIMILARITY = False  # whether or not self similarity should be computed

#############################
# processed variables and constants and instantiations

print(f'Rasteranalysis in {MODE} mode')
params = parameters()
utils = utilities()

# collect all discontinuity ids
ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]

#############################
# main loop

processed_sets = 0
while processed_sets < N_SETS_TO_PROCESS:

    already_processed = [c.split('_')[0] for c in listdir(r'../combinations') if '_RasterAnalysis' in c]
    ids_unprocessed = np.where(np.in1d(ids, already_processed) == False)[0]

    if MODE == 'random':
        set_id = np.random.choice(ids_unprocessed, size=1)[0]
    elif MODE == 'sequential':
        set_id = ids_unprocessed[0]

    name = f'{ids[set_id]}_RasterAnalysis.txt'

    print(f'\nprocessing set {ids[set_id]}')
    discontinuities = o3d.io.read_triangle_mesh(fr'../combinations/{ids[set_id]}_discontinuities.stl')
    print('\tdiscontinuities loaded')

    boxes_mesh = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        input=discontinuities, voxel_size=voxel_size,
        min_bound=np.array([0, 0, 0]),
        max_bound=np.array([TOT_BBOX_SIZE, TOT_BBOX_SIZE, TOT_BBOX_SIZE]))
    del discontinuities
    print('\tdiscontinuity voxels created')

    boxes_all = o3d.geometry.VoxelGrid.create_dense(
        width=TOT_BBOX_SIZE, height=TOT_BBOX_SIZE, depth=TOT_BBOX_SIZE,
        voxel_size=voxel_size, origin=np.array([0, 0, 0]),
        color=[color, color, color])
    print('\toverall voxels created')

    combined = boxes_mesh + boxes_all
    del boxes_mesh
    del boxes_all
    gc.collect()

    # convert voxels to 3D numpy array
    gridded_voxels = utils.voxel2grid(combined, RESOLUTION, color)
    gc.collect()

    # compute structural complexity acc. to Bagrov et al. (2020)
    c = params.structural_complexity(gridded_voxels, mode='3Dgrid')
    print(f'\tstructural complexity: {c}')

    if COMPUTE_SIMILARITY is True:
        # compute number of voxels per unit box, rounded down to avoid boundary
        # issues
        resolution = int(UNIT_BOX_SIZE/voxel_size)
        intervalls = np.arange(0, TOT_BBOX_SIZE*resolution, resolution)

        # count how many blocks are similar to each other
        similarity_count = []
        for x1 in intervalls:
            for y1 in intervalls:
                for z1 in intervalls:
                    test_block1 = gridded_voxels[x1:x1+resolution,
                                                 y1:y1+resolution,
                                                 z1:z1+resolution]
                    dices = []
                    for x2 in intervalls:
                        for y2 in intervalls:
                            for z2 in intervalls:

                                test_block2 = gridded_voxels[x2:x2+resolution,
                                                             y2:y2+resolution,
                                                             z2:z2+resolution]
                                # n elements same in array 1 and 2
                                n_same = (test_block1 == test_block2).sum()
                                # compute dice coefficient of similarity
                                dice = (2*n_same) / (resolution**3 + resolution**3)
                                dices.append(dice)
                    # -1 due to the perfect self similarity
                    similarity_count.append(len(np.where(np.array(dices) > DICE_THRESH)[0])-1)

        n_zeros = len(np.where(np.array(similarity_count) == 0)[0])
        max_ = max(similarity_count)
        min_ = min(similarity_count)
        mean = np.mean(similarity_count)
        med = np.median(similarity_count)
        del similarity_count
    else:
        n_zeros, max_, min_, mean, med = np.nan, np.nan, np.nan, np.nan, np.nan

    del gridded_voxels
    gc.collect()

    with open(fr'../combinations/{name}', 'w') as f:
        f.write(f'{n_zeros},{max_},{min_},{mean},{med},{c}')

    print(f'\t{name} finished')
    processed_sets += 1

    # check if all files were processed
    n_finished = [d for d in listdir(r'../combinations') if '_RasterAnalysis' in d]
    if len(n_finished) == len(ids):
        print('!!ALL FILES PROCESSED!!')
        break
