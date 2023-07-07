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
# static variables and constants

N_SETS_TO_PROCESS = 400  # max number of sets to process in this run
TOT_BBOX_SIZE = 10  # size of full bounding box
# run code for random- or sequential unprocessed samples -> multiprocessing
MODE = 'random'  # 'random', 'sequential'

#############################
# main loop

print(f'Boxcounting in {MODE} mode')

processed_sets = 0
while processed_sets < N_SETS_TO_PROCESS:
    # collect all discontinuity ids
    ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]
    already_processed = [c.split('_')[0] for c in listdir(r'../combinations') if '_boxcount' in c]
    ids_unprocessed = np.where(np.in1d(ids, already_processed) == False)[0]

    if MODE == 'random':
        set_id = np.random.choice(ids_unprocessed, size=1)[0]
    elif MODE == 'sequential':
        set_id = ids_unprocessed[0]

    name = f'{ids[set_id]}_boxcount.txt'

    print(f'processing set {ids[set_id]}\n')
    discontinuities = o3d.io.read_triangle_mesh(fr'../combinations/{ids[set_id]}_discontinuities.stl')
    print('discontinuities loaded')

    #############################
    # collect intersections
    n_boxes = []
    n_boxes_theo = []
    box_sizes = []

    for i, box_size in enumerate([0.25, 0.125, 0.0625, 0.03125]):
        start = datetime.now()

        print(f'create new voxel grid with box size {box_size}')
        boxes = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            input=discontinuities, voxel_size=box_size,
            min_bound=np.array([0, 0, 0]),
            max_bound=np.array([TOT_BBOX_SIZE, TOT_BBOX_SIZE, TOT_BBOX_SIZE]))

        # convert voxels to pandas dataframe for further analysis
        points = [np.array(boxes.get_voxel_center_coordinate(v.grid_index)) for v in boxes.get_voxels()]

        df = pd.DataFrame(columns=['X', 'Y', 'Z'], data=np.array(points))
        # cut away wrongly created out of bounds voxels
        df = df[(df['X'] >= 0) & (df['X'] <= TOT_BBOX_SIZE)]
        df = df[(df['Y'] >= 0) & (df['Y'] <= TOT_BBOX_SIZE)]
        df = df[(df['Z'] >= 0) & (df['Z'] <= TOT_BBOX_SIZE)]

        # collect number of boxes that would fill up whole space
        n_theo = int((TOT_BBOX_SIZE / box_size)**3)
        n_boxes_theo.append(n_theo)

        # measure actually needed number of boxes
        required_boxes = len(df)

        n_boxes.append(required_boxes)
        box_sizes.append(box_size)
        print(f'\t{n_theo}, required: {required_boxes}')

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
    n_finished = [d for d in listdir(r'../combinations') if 'boxcount' in d]
    if len(n_finished) == len(ids):
        print('!!ALL FILES PROCESSED!!')
        break
