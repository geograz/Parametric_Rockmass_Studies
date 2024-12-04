# -*- coding: utf-8 -*-
"""
Code to the paper "Rock mass structure characterization considering finite and
folded discontinuities"
Dr. Georg H. Erharter - 2023
DOI: https://doi.org/10.1007/s00603-024-03787-9

Script that compiles the recorded data from samples of the discrete
discontinuity networks and creates one excel file for further processing.
"""

import gzip
import numpy as np
from os import listdir
import pandas as pd
import pickle
from tqdm import tqdm


##########################################
# static variables and constants
##########################################

RASTER_RESOLUTIONS = [0.25, 0.2, 0.15, 0.1, 0.05]

INPUTS = ['bounding box size [m]', 'Jv boxes edge size [m]', 'seed',
          'set 1 - n joints', 'set 1 - radius [m]', 'set 1 - radius std [m]',
          'set 1 - dip direction [°]', 'set 1 - dip direction std [°]',
          'set 1 - dip [°]', 'set 1 - dip std [°]', 'set 1 - type',
          'F_rand_sin', 'F_rand_n_planes', 'F_rand_angle', 'F_rand_axis_x',
          'F_rand_axis_y', 'F_rand_axis_z', 'set 2 - n joints',
          'set 2 - radius [m]', 'set 2 - radius std [m]',
          'set 2 - dip direction [°]', 'set 2 - dip direction std [°]',
          'set 2 - dip [°]', 'set 2 - dip std [°]', 'set 3 - n joints',
          'set 3 - radius [m]', 'set 3 - radius std [m]',
          'set 3 - dip direction [°]', 'set 3 - dip direction std [°]',
          'set 3 - dip [°]', 'set 3 - dip std [°]', 'random set - n joints',
          'random set - radius [m]', 'random set - radius std [m]',
          'identifier']

OUTPUTS = ['meas. spacing set 1 [m]', 'meas. spacing set 2 [m]',
           'meas. spacing set 3 [m]', 'RQD Y', 'RQD X', 'RQD Z',
           'apparent spacing Y [m]', 'apparent spacing X [m]',
           'apparent spacing Z [m]', 'P10 Y', 'P10 X', 'P10 Z', 'P20 X',
           'P21 X', 'P20 Y', 'P21 Y', 'P20 Z', 'P21 Z',
           'Jv measured [discs/m³]', 'P32', 'n blocks',
           'avg. block volume [m³]', 'max block volume [m³]',
           'min block volume [m³]', 'avg. block edge length [m]',
           'avg. block surface area [m²]', 'a3', 'a2', 'a1',
           'set 1 total area [m2]', 'set 2 total area [m2]',
           'set 3 total area [m2]', 'random set total area [m2]']

##########################################
# main dataset compilation
##########################################

# load text files with geometry parameters as they are saved by grasshopper
print('loading data from textfiles')
contents = []
for file_name in tqdm(listdir(r'../combinations')):
    if '.txt' in file_name:
        f = open(fr'../combinations/{file_name}', 'r')
        content = f.read()
        content = content.replace('(', '')
        content = content.replace(')', '')
        content = content.replace(' ', '')
        content = content.replace('L', '')
        content = [eval(num) for num in content.split(',')]
        contents.append(content)
        f.close()

# make pandas dataframe
columns = INPUTS + OUTPUTS
df = pd.DataFrame(columns=columns, data=np.array(contents))

# remove unused input data from discontinuity set 1 which is either a set of
# planar, finite discontinuities or folded discontinuities
id_plane = np.where(df['set 1 - type'] == 0)[0]
id_folds = np.where(df['set 1 - type'] == 1)[0]

df.loc[id_folds, ['set 1 - n joints', 'set 1 - radius [m]',
                  'set 1 - radius std [m]', 'set 1 - dip direction [°]',
                  'set 1 - dip direction std [°]', 'set 1 - dip [°]',
                  'set 1 - dip std [°]']] = np.nan
df.loc[id_plane, ['F_rand_sin', 'F_rand_n_planes', 'F_rand_angle',
                  'F_rand_axis_x', 'F_rand_axis_y', 'F_rand_axis_z']] = np.nan
df.set_index('identifier', inplace=True)

print('basic data frame set up\n')

##########################################
# add voxel counts where they are computed

# set up empty columns for population with voxel data
cols_empt_voxels = [f'n empty voxels at {r} [m]' for r in RASTER_RESOLUTIONS]
cols_disc_voxels = [f'n disc. voxels at {r} [m]' for r in RASTER_RESOLUTIONS]
for col in cols_empt_voxels + cols_disc_voxels:
    df[col] = np.nan

print('loading voxel data')
for sample in tqdm(df.index):
    for resolution in RASTER_RESOLUTIONS:
        fp = fr'..\rasters\{sample}_{resolution}.pkl.gz'
        try:
            with gzip.open(fp, 'rb') as f:
                decompressed_voxel_array = pickle.load(f)
            n_empty_voxels = len(np.where(decompressed_voxel_array == 0)[0])
            n_disc_voxels = len(np.where(decompressed_voxel_array == 1)[0])
            df.loc[sample,
                   f'n empty voxels at {resolution} [m]'] = n_empty_voxels
            df.loc[sample,
                   f'n disc. voxels at {resolution} [m]'] = n_disc_voxels
            # safety check to see if all values are 0 or 1
            if decompressed_voxel_array.size != n_empty_voxels + n_disc_voxels:
                raise ValueError(f'array of {sample} contains non binary vals')
        except FileNotFoundError:
            pass
print('voxel data added\n')

##########################################
# some basic data preprocessing

# convert object dtypes to float
for param in df.columns:
    df[param] = pd.to_numeric(df[param])

# replace joint properties for joint sets that have no joints
for joint_set in [1, 2, 3, 4]:
    if joint_set == 4:
        id_0 = np.where(df['random set - n joints'] == 0)[0]
        id_0 = df.index[id_0]
        df.loc[id_0, 'random set - radius [m]'] = np.nan
    else:
        id_0 = np.where(df[f'set {joint_set} - n joints'] == 0)[0]
        id_0 = df.index[id_0]
        df.loc[id_0, f'set {joint_set} - radius [m]'] = np.nan
        df.loc[id_0, f'set {joint_set} - dip [°]'] = np.nan
        df.loc[id_0, f'set {joint_set} - dip direction [°]'] = np.nan

# idx_neg_vol = np.where(df['avg. block volume [m³]'] < 0)[0]
# df = df.drop(index=idx_neg_vol)

# drop all columns of features that are still under development such as block
# volumes or fractal dimensions
df.drop(['n blocks', 'avg. block volume [m³]', 'max block volume [m³]',
         'min block volume [m³]', 'avg. block edge length [m]',
         'avg. block surface area [m²]', 'a3', 'a2', 'a1'],
        axis=1, inplace=True)

# save to excel file
df.to_excel(r'../output/PDD1_0.xlsx')
print('data compiled and saved')
