# -*- coding: utf-8 -*-
"""
        PARAMETRIC ROCK MASS STUDIES
-- computational rock mass characterization --

Code author: Dr. Georg H. Erharter

Script that loads meshes and computes rasters at different resolution from them
for further analyses. Does not include raster analyses - only generation.
"""

from os import listdir

import gzip
import pickle
import gc
import numpy as np
import pandas as pd
import trimesh

from X_library import parameters, utilities

#############################
# static variables and constants

N_SETS_TO_PROCESS = 3000  # max number of sets to process in this run
RASTER_RESOLUTIONS = [0.25, 0.2, 0.15, 0.1, 0.05]  # 3D grid resolution
MAX_RESOLUTION = 0.05  # max. resolution to process in current run
SAVE_CSV = False  # convert rastered discontinuity array to pointcloud & save
SAVE_ZIP = True  # save rastered discontinuity array as zip file
# run code for random- or sequential unprocessed samples -> multiprocessing
MODE = 'sequential'  # 'random', 'sequential' 'sequential neg'
FP_DF_MEMORY_ERROR = r'../output/memory_errors.xlsx'  # excel for memory errors
FP_DF_SAMPLES = r'../output/df_samples.csv'

#############################
# processed variables and instantiations

print(f'Raster generation in {MODE} mode with max resolution {MAX_RESOLUTION}')
params = parameters()
utils = utilities()

# load or make dataframe with all samples
try:
    df_samples = pd.read_csv(FP_DF_SAMPLES)
except FileNotFoundError:
    # collect all discontinuity ids
    ids = [c.split('_')[0] for c in listdir(r'../combinations') if 'discontinuities' in c]
    # generate raster names and resolutions
    all_rasters, all_resolutions = [], []
    for id_ in ids:
        for res in RASTER_RESOLUTIONS:
            all_rasters.append(f'{id_}_{res}')
            all_resolutions.append(res)
    df_samples = pd.DataFrame({'Sample ID': all_rasters,
                               'Sample resolution [m]': all_resolutions,
                               'state': np.zeros(len(all_rasters)).astype(np.int16)})
    # states: 0 = unprocessed, 1 = processed, 2 = memory error
    # set states for already processed ones & memory errors
    already_processed = [p.replace('.pkl.gz', '') for p in listdir(r'../rasters')]
    ids_processed = np.where(
        np.isin(df_samples['Sample ID'],
                already_processed) == True)[0]
    df_samples.loc[ids_processed, 'state'] = 1
    # load old memory errors
    df_mem_err = pd.read_excel(FP_DF_MEMORY_ERROR)
    ids_mem_err = np.where(
        np.isin(df_samples['Sample ID'],
                list(df_mem_err['sample ID'])) == True)[0]
    df_samples.loc[ids_mem_err, 'state'] = 2
    df_samples.to_csv(FP_DF_SAMPLES, index=False)

#############################
# main loop

processed_sets = 0  # counter
while processed_sets < N_SETS_TO_PROCESS:

    # load df that keeps track of samples states
    df_samples = pd.read_csv(FP_DF_SAMPLES)
    # get unprocessed samples of right resolution
    unp = df_samples[df_samples['state'] == 0]
    unp_res = unp[unp['Sample resolution [m]'] >= MAX_RESOLUTION]
    # check if all files were processed
    if len(unp_res) == 0:
        print('!!ALL FILES PROCESSED!!')
        break

    # choose id to process dependent on mode
    if MODE == 'random':
        id_ = np.random.choice(unp_res['Sample ID'].index, size=1)[0]
    elif MODE == 'sequential':  # begin rasterization from start
        id_ = unp_res['Sample ID'].index[0]
    elif MODE == 'sequential neg':  # begin rasterization from end
        id_ = unp_res['Sample ID'].index[-1]
    name, resolution = unp_res.loc[id_, ['Sample ID', 'Sample resolution [m]']]

    print(f'\nprocessing set {name} at resolution: {resolution}')
    print(f'\t{len(unp_res)} samples unprocessed')
    # load mesh
    fp = fr"..\combinations\{name.split('_')[0]}_discontinuities.stl"
    discontinuity_mesh = trimesh.load_mesh(fp)
    print('\tdiscontinuity mesh loaded')
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
        # update sample overview
        state = 1
        print(f'\t{name} finished')
    # some meshes are too complex for detailed rasterization
    except MemoryError:
        # update sample overview
        state = 2
        print(f'\t{name} failed due to memory error')
        pass
    # update and save sample overview dataframe
    df_samples = pd.read_csv(FP_DF_SAMPLES)
    df_samples.loc[id_, 'state'] = state
    df_samples.to_csv(FP_DF_SAMPLES, index=False)

    processed_sets += 1
    print(f'\t{processed_sets}/{N_SETS_TO_PROCESS} sets processed this run')
