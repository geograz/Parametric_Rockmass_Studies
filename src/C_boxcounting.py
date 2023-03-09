# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:02:53 2023

@author: Schorsch
"""

from datetime import datetime
import numpy as np
import open3d as o3d
from os import listdir
import pandas as pd

#############################
# function definitions

N_SETS_TO_PROCESS = 100  # max number of sets to process in this run
TOT_BBOX_SIZE = 10  # size of full bounding box
M_BOX_SIZE = 1  # size of initial fractal measurement box
N_SPLITS_REQUIRED = 4


processed_sets = 0
while processed_sets < N_SETS_TO_PROCESS:

    ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]
    set_id = np.random.randint(0, len(ids))
    name = f'{ids[set_id]}_boxcount.txt'

    if name in listdir(r'../combinations'):
        print(f'{name} already processed -> skip')
        pass
    else:
        print(f'processing set {ids[set_id]}\n')
        discontinuities = o3d.io.read_triangle_mesh(fr'../combinations/{ids[set_id]}_discontinuities.stl')

        # set number of initial splits
        split = TOT_BBOX_SIZE / M_BOX_SIZE  # initial split
        # create initial voxel grid
        box_size = M_BOX_SIZE  # initial box edge length

        #############################
        # iterate the splits and collect intersections
        n_boxes = []
        n_boxes_theo = []
        box_sizes = []

        i = 0  # iterations counter

        for box_size in [0.25, 0.125, 0.0625, 0.03125]:
            start = datetime.now()

            print(f'create new voxel grid with box size {box_size}')
            boxes = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                input=discontinuities, voxel_size=box_size,
                min_bound=np.array([0, 0, 0]),
                max_bound=np.array([TOT_BBOX_SIZE, TOT_BBOX_SIZE, TOT_BBOX_SIZE]))

            # collect number of boxes that would fill up whole space
            n_theo = int((TOT_BBOX_SIZE / box_size)**3)
            n_boxes_theo.append(n_theo)

            # measure actually needed number of boxes and create new ones
            required_boxes = len(boxes.get_voxels())

            n_boxes.append(required_boxes)
            box_sizes.append(box_size)
            print(f'\t{n_theo}, required: {required_boxes}')

            i += 1  # update counter
            stop = datetime.now()
            delta = (stop - start).total_seconds()
            print(f'\tsplit {i} done in {round(delta, 2)} s')

        #############################
        # save boxes

        n_boxes, box_sizes = np.array(n_boxes), np.array(box_sizes)

        df = pd.DataFrame({'n boxes': n_boxes,
                           'box edge length [m]': box_sizes})
        df.to_csv(fr'../combinations/{name}', index=False)

        print(f'set {ids[set_id]} finished \n')
        processed_sets += 1

    # check if all files were processed
    n_finished = [d for d in listdir(r'../combinations') if 'fractals' in d]
    if len(n_finished) == len(ids):
        print('!!ALL FILES PROCESSED!!')
        break
