# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 09:12:18 2023

@author: GEr
"""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from os import listdir
import pandas as pd

from X_library import parameters

params = parameters()

N_SETS_TO_PROCESS = 100  # max number of sets to process in this run

TOT_BBOX_SIZE = 10  # total bounding box size [m]
UNIT_BOX_SIZE = 1  # size of a measurement box [m]
RESOLUTION = 256  # 3D grid resolution
voxel_size = TOT_BBOX_SIZE / RESOLUTION  # .05
color = 1
DICE_THRESH = 0.75  # threshold of dice coefficient that indicates similarity

processed_sets = 0
while processed_sets < N_SETS_TO_PROCESS:
    ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]
    set_id = np.random.randint(0, len(ids))
    name = f'{ids[set_id]}_RasterAnalysis.txt'

    if name in listdir(r'../combinations'):
        print(f'{name} already processed -> skip')
        pass
    else:
        print(name)
        name = name.split('_')[0]
        discontinuities = o3d.io.read_triangle_mesh(fr'../combinations/{name}_discontinuities.stl')
        print('\tdiscontinuities loaded')

        boxes_all = o3d.geometry.VoxelGrid.create_dense(width=TOT_BBOX_SIZE,
                                                        height=TOT_BBOX_SIZE,
                                                        depth=TOT_BBOX_SIZE,
                                                        voxel_size=voxel_size,
                                                        origin=np.array([0, 0, 0]),
                                                        color=[color, color, color])
        print('\toverall voxels created')

        boxes_mesh = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            input=discontinuities, voxel_size=voxel_size,
            min_bound=np.array([0, 0, 0]),
            max_bound=np.array([TOT_BBOX_SIZE, TOT_BBOX_SIZE, TOT_BBOX_SIZE]))
        print('\tdiscontinuity voxels created')

        combined = boxes_mesh + boxes_all

        points = []
        for v in combined.get_voxels():
            coords = combined.get_voxel_center_coordinate(v.grid_index)
            if sum(v.color) == color*3:
                intersecting = -1  # non intersecting voxel
            else:
                intersecting = 1  # intersecting voxel
            points.append(np.hstack((coords, intersecting)))

        df = pd.DataFrame(columns=['X', 'Y', 'Z', 'inters'], data=np.array(points))

        # convert voxels to grid without XYZ coords
        side_length = int(round(len(df)**(1/3), 0))
        # print(f'\t{side_length}')
        idx = np.lexsort((df['X'], df['Y'], df['Z'])).reshape(side_length, side_length, side_length)
        gridded_voxels = df['inters'].values[idx]

        # compute structural complexity acc. to Bagrov et al. (2020)
        c = params.structural_complexity(gridded_voxels, mode='3Dgrid')
        print(f'\tstructural complexity: {c}')

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
                                dice = (2*n_same) / (resolution**3 + resolution**3)
                                dices.append(dice)
                    # -1 due to the perfect self similarity
                    similarity_count.append(len(np.where(np.array(dices) > DICE_THRESH)[0])-1)

        # fig, ax = plt.subplots()
        # ax.hist(similarity_count, bins=50, edgecolor='black')
        # plt.tight_layout()

        # fig, ax = plt.subplots()
        # ax.imshow(gridded_voxels[:,:,2])
        # plt.tight_layout()

        n_zeros = len(np.where(np.array(similarity_count) == 0)[0])
        max_ = max(similarity_count)
        min_ = min(similarity_count)
        mean = np.mean(similarity_count)
        med = np.median(similarity_count)

        with open(fr'../combinations/{name}_RasterAnalysis.txt', 'w') as f:
            f.write(f'{n_zeros},{max_},{min_},{mean},{med},{c}')

        print(f'\t{name} finished')
        processed_sets += 1
        # df.to_csv('test.csv', index=False)
        # df.to_parquet('test.gzip', engine='pyarrow', index=False)
    # check if all files were processed
    n_finished = [d for d in listdir(r'../combinations') if '_RasterAnalysis' in d]
    if len(n_finished) == len(ids):
        print('!!ALL FILES PROCESSED!!')
        break
