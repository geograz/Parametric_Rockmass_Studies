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

N_SETS_TO_PROCESS = 15  # max number of sets to process in this run
TOT_BBOX_SIZE = 10  # total bounding box size [m]
RESOLUTIONS = [0.25, 0.2, 0.15, 0.1, 0.05]  # 3D grid resolution
SAVE_CSV = False
SAVE_ZIP = True
# run code for random- or sequential unprocessed samples -> multiprocessing
MODE = 'sequential'  # 'random', 'sequential'

#############################
# processed variables and constants and instantiations

print(f'Raster generation in {MODE} mode')
params = parameters()
utils = utilities()

# collect all discontinuity ids
ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]
names = []
for id_ in ids:
    for res in RESOLUTIONS:
        names.append(f'{id_}_{res}')

#############################
# main loop

# TODO set code up so that it only creates rasters at different resolutions
# TODO make separate script for raster analysis including complexity measures & fractal dimensions

processed_sets = 0
while processed_sets < N_SETS_TO_PROCESS:

    already_processed = [ap.replace('.pkl.gz', '') for ap in listdir(r'../rasters') if '.pkl.gz' in ap]

    ids_unprocessed = np.where(np.isin(names, already_processed) == False)[0]

    if MODE == 'random':
        set_id = np.random.choice(ids_unprocessed, size=1)[0]
    elif MODE == 'sequential':
        set_id = ids_unprocessed[0]

    name = names[set_id]

    print(f'\nprocessing set {name}')
    # load mesh
    fp = fr"..\combinations\{name.split('_')[0]}_discontinuities.stl"
    discontinuity_mesh = trimesh.load_mesh(fp)
    print('\tdiscontinuity mesh loaded')
    # for resolution in RESOLUTIONS:
    resolution = eval(name.split('_')[1])
    print(f'\tprocessing resolution {resolution}')
    try:
        discontinuity_voxels = discontinuity_mesh.voxelized(pitch=resolution)
        del discontinuity_mesh
        gc.collect()
        print('\tdiscontinuity voxels created')
        discontinuity_array = discontinuity_voxels.matrix.astype(int)
        del discontinuity_voxels
        gc.collect()
        print('\tdiscontinuity grid created')
    
        if SAVE_CSV is True:
            n = discontinuity_array.shape[0]
            x = np.arange(0, n * resolution, resolution)
            y = np.arange(0, n * resolution, resolution)
            z = np.arange(0, n * resolution, resolution)
            # Create the 3D grid of coordinates
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            x_flat = X.ravel()
            y_flat = Y.ravel()
            z_flat = Z.ravel()
            data_flat = discontinuity_array.ravel()
            output = np.column_stack((x_flat, y_flat, z_flat, data_flat))
    
            df = pd.DataFrame(data=output, columns=['x', 'y', 'z', 'v'])
            df = df.astype({'x': 'float16', 'y': 'float16', 'z': 'float16',
                            'v': 'int8'})
            df.to_csv(fr'../rasters/{name}.csv', index=False)
            print('\tcsv voxels saved')
    
        if SAVE_ZIP is True:
            with gzip.open(fr'../rasters/{name}.pkl.gz', 'wb') as f:
                pickle.dump(discontinuity_array, f)
            print('\tzip voxels saved')
    
        # # compute structural complexity acc. to Bagrov et al. (2020)
        # c = params.structural_complexity(discontinuity_array, mode='3Dgrid')
        # del discontinuity_array
        # gc.collect()
        # print(f'\tstructural complexity: {round(c, 3)}')
    
        # with open(fr'../combinations/{name}', 'w') as f:
        #     f.write(f'{c}')
        print(f'\t{name} finished')
    except MemoryError:
        pass

    processed_sets += 1

    # check if all files were processed
    n_finished = listdir(r'../rasters')
    if len(n_finished) == len(ids):
        print('!!ALL FILES PROCESSED!!')
        break
