# -*- coding: utf-8 -*-
"""
        PARAMETRIC ROCK MASS STUDIES
-- computational rock mass characterization --

Code author: Dr. Georg H. Erharter

Script that loads meshes and computes rasters at different resolution from them
for further analyses. Does not include raster analyses - only generation.
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

N_SETS_TO_PROCESS = 4000  # max number of sets to process in this run
RASTER_RESOLUTIONS = [0.25, 0.2, 0.15, 0.1, 0.05]  # 3D grid resolution
SAVE_CSV = False  # convert rastered discontinuity array to pointcloud & save
SAVE_ZIP = True  # save rastered discontinuity array as zip file
# run code for random- or sequential unprocessed samples -> multiprocessing
MODE = 'sequential'  # 'random', 'sequential'
FP_DF_MEMORY_ERROR = r'../output/memory_errors.xlsx'  # excel for memory errors

#############################
# processed variables and constants and instantiations

print(f'Raster generation in {MODE} mode')
params = parameters()
utils = utilities()

# collect all discontinuity ids
ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]
names = []
for id_ in ids:
    for res in RASTER_RESOLUTIONS:
        names.append(f'{id_}_{res}')

#############################
# main loop

try:  # to load existing memory errors
    df_memory_errors = pd.read_excel(FP_DF_MEMORY_ERROR)
except FileNotFoundError:
    df_memory_errors = pd.DataFrame(columns=['sample ID'])
    df_memory_errors.to_excel(FP_DF_MEMORY_ERROR, index=False)
failed = list(df_memory_errors['sample ID'].values)

processed_sets = 0  # counter
while processed_sets < N_SETS_TO_PROCESS:
    # check which samples have not yet been processed
    already_processed = [ap.replace('.pkl.gz', '') for ap in listdir(r'../rasters') if '.pkl.gz' in ap] + failed
    ids_unprocessed = np.where(np.isin(names, already_processed) == False)[0]

    # choose id to process dependent on mode
    if MODE == 'random':
        set_id = np.random.choice(ids_unprocessed, size=1)[0]
    elif MODE == 'sequential':
        set_id = ids_unprocessed[processed_sets]
    name = names[set_id]

    print(f'\nprocessing set {name}')
    # load mesh
    fp = fr"..\combinations\{name.split('_')[0]}_discontinuities.stl"
    discontinuity_mesh = trimesh.load_mesh(fp)
    print('\tdiscontinuity mesh loaded')
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
        # save options
        if SAVE_CSV is True:
            utils.array_to_pointcloud(
                discontinuity_array, resolution,
                savepath=fr'../rasters/{name}.zip')
            print('\tcsv voxels saved')
        if SAVE_ZIP is True:
            with gzip.open(fr'../rasters/{name}.pkl.gz', 'wb') as f:
                pickle.dump(discontinuity_array, f)
            print('\tzip voxels saved')

        print(f'\t{name} finished')
    # some meshes are too complex for detailed rasterization
    except MemoryError:
        df_memory_errors = pd.read_excel(FP_DF_MEMORY_ERROR)
        df_memory_errors.loc[len(df_memory_errors), 'sample ID'] = name
        failed = list(df_memory_errors['sample ID'].values)
        df_memory_errors.to_excel(FP_DF_MEMORY_ERROR, index=False)
        pass

    processed_sets += 1
    print(f'\t{processed_sets}/{N_SETS_TO_PROCESS} sets processed this run')

    # check if all files were processed
    n_finished = listdir(r'../rasters')
    if len(n_finished) >= len(ids)*len(RASTER_RESOLUTIONS):
        print('!!ALL FILES PROCESSED!!')
        break
