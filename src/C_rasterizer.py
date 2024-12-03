# -*- coding: utf-8 -*-
"""
Experimental code to the paper XXXX
Dr. Georg H. Erharter - 2023
DOI: https://doi.org/10.1007/s00603-024-03787-9

Script that loads meshes and transforms them into rasters at different
resolution for further analyses.
(results not included in first paper).
"""
import gzip
import pickle
import gc
import numpy as np
import pandas as pd
import trimesh
from os import listdir

from X_library import parameters, utilities

#############################
# static variables and constants

N_SETS_TO_PROCESS = 2  # max number of sets to process in this run
TOT_BBOX_SIZE = 10  # total bounding box size [m]
RESOLUTION = 0.2  # 3D grid resolution
SAVE_FILES = True
# run code for random- or sequential unprocessed samples -> multiprocessing
MODE = 'sequential'  # 'random', 'sequential'

#############################
# processed variables and constants and instantiations

print(f'Rasteranalysis in {MODE} mode')
params = parameters()
utils = utilities()

# collect all discontinuity ids
ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]

#############################
# main loop

# TODO set code up so that it only creates rasters at different resolutions
# TODO make separate script for raster analysis including complexity measures & fractal dimensions

processed_sets = 0
while processed_sets < N_SETS_TO_PROCESS:

    already_processed = [c.split('_')[0] for c in listdir(r'../combinations') if '_complexity' in c]
    ids_unprocessed = np.where(np.isin(ids, already_processed) == False)[0]

    if MODE == 'random':
        set_id = np.random.choice(ids_unprocessed, size=1)[0]
    elif MODE == 'sequential':
        set_id = ids_unprocessed[0]

    name = f'{ids[set_id]}_complexity.txt'

    print(f'\nprocessing set {ids[set_id]}')

    fp = fr'../combinations/{ids[set_id]}_discontinuities.stl'
    discontinuity_mesh = trimesh.load_mesh(fp)
    print('\tdiscontinuities loaded')
    discontinuity_voxels = discontinuity_mesh.voxelized(pitch=RESOLUTION)
    del discontinuity_mesh
    gc.collect()
    print('\tdiscontinuity voxels created')
    discontinuity_array = discontinuity_voxels.matrix.astype(int)
    del discontinuity_voxels
    gc.collect()
    print('\tdiscontinuity grid created')
    if SAVE_FILES is True:
        occupied_indices = np.array(np.where(discontinuity_array == 1)).T
        unoccupied_indices = np.array(np.where(discontinuity_array == 0)).T
        # Convert indices to world coordinates
        occupied_coords = occupied_indices * RESOLUTION
        unoccupied_coords = unoccupied_indices * RESOLUTION

        df_occupied = pd.DataFrame(data=occupied_coords,
                                   columns=['x', 'y', 'z'])
        df_occupied['v'] = 1
        df_unoccupied = pd.DataFrame(data=unoccupied_coords,
                                     columns=['x', 'y', 'z'])
        df_unoccupied['v'] = 0
        df = pd.concat((df_occupied, df_unoccupied))
        df = df.astype({'x': 'float16', 'y': 'float16', 'z': 'float16',
                        'v': 'int8'})
        df.to_csv(fr'../combinations/{ids[set_id]}_voxels.csv', index=False)

        with gzip.open(fr'../combinations/{ids[set_id]}_voxel_array.pkl.gz', 'wb') as f:
            pickle.dump(discontinuity_array, f)

        print('\tvoxels saved')

    # compute structural complexity acc. to Bagrov et al. (2020)
    c = params.structural_complexity(discontinuity_array, mode='3Dgrid')
    del discontinuity_array
    gc.collect()
    print(f'\tstructural complexity: {round(c, 3)}')

    with open(fr'../combinations/{name}', 'w') as f:
        f.write(f'{c}')

    print(f'\t{name} finished')
    processed_sets += 1

    # check if all files were processed
    n_finished = [d for d in listdir(r'../combinations') if '_complexity' in d]
    if len(n_finished) == len(ids):
        print('!!ALL FILES PROCESSED!!')
        break
