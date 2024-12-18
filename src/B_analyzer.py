# -*- coding: utf-8 -*-
"""
        PARAMETRIC ROCK MASS STUDIES
-- computational rock mass characterization --

Code author: Dr. Georg H. Erharter

Script that processes the compiled records of the discrete discontinuity
dataset, computes new parameters and creates figures to visualize the dataset.
"""

import gzip
import numpy as np
import pandas as pd
import pickle
from skimage.measure import euler_number
from tqdm import tqdm

from X_library import math, parameters, utilities


##########################################
# static variables and constants
##########################################

RASTER_RESOLUTIONS = [0.25, 0.2, 0.15, 0.1, 0.05]
DO_BLOCK_ANALYSES = False
SAVE_BLOCKS = False  # whether or not rastered block models should be saved

# new parameter names that will be computed
COMPLEXITY_COLUMNS = ['Minkowski dimension', 'structural complexity',
                      'compression ratio', 'Euler characteristic',
                      'Euler characteristic inverted']
BLOCK_COLUMNS = ['n blocks', 'avg. block volume [m3]',
                 'median block volume [m3]']

##########################################
# instantiations, data loading and preprocessing
##########################################

m = math()
params = parameters()
utils = utilities()

pd.options.mode.chained_assignment = None

df = pd.read_excel(r'../output/PDD1_1.xlsx', index_col='identifier')

id_JV_max = df['Jv measured [discs/m³]'].argmax()
id_JV_min = df['Jv measured [discs/m³]'].argmin()

print(f'sample with lowest JV: {df.index[id_JV_min]}')
print(f'sample with highest JV: {df.index[id_JV_max]}')

##########################################
# compute "classical" rock engineering parameters
##########################################

# compute average values of directionally dependent measurements
df['avg. P10'] = np.mean(df[['P10 Y', 'P10 X', 'P10 Z']].values, axis=1)
df['avg. P20'] = np.mean(df[['P20 Y', 'P20 X', 'P20 Z']].values, axis=1)
df['avg. P21'] = np.mean(df[['P21 Y', 'P21 X', 'P21 Z']].values, axis=1)
df['avg. app. spacing [m]'] = np.mean(df[['apparent spacing Y [m]',
                                          'apparent spacing X [m]',
                                          'apparent spacing Z [m]']].values,
                                      axis=1)
df['avg. RQD'] = np.mean(df[['RQD Y', 'RQD X', 'RQD Z']].values, axis=1)

# compute different versions of the volumetric joint count acc. to literature
df['Jv ISO 14689 (2019)'] = params.Jv_ISO14689(
    df['meas. spacing set 1 [m]'], df['meas. spacing set 2 [m]'],
    df['meas. spacing set 3 [m]'])

df['Jv Palmstrøm (2000)'] = params.Jv_Palmstroem2000(
    df['meas. spacing set 1 [m]'], df['meas. spacing set 2 [m]'],
    df['meas. spacing set 3 [m]'], df['random set - n joints'],
    df['bounding box size [m]']**3)

df['Jv Sonmez & Ulusay (1999) 1'] = params.Jv_Sonmez1999_1(df['P10 X'],
                                                           df['P10 Y'],
                                                           df['P10 Z'])

df['Jv Sonmez & Ulusay (1999) 2'] = params.Jv_Sonmez1999_2(df['avg. P10'])

df['Jv Erharter (2023)'] = 3*df['avg. P10'] * 0.8 + 2

# compute set ratio based on discontinuity area per set to determine the number
# of discontinuity sets
df['tot disc. area [m2]'] = df[['set 1 total area [m2]',
                                'set 2 total area [m2]',
                                'set 3 total area [m2]',
                                'random set total area [m2]']].sum(axis=1)
df['set_1_ratio'] = df['set 1 total area [m2]'] / df['tot disc. area [m2]']
df['set_2_ratio'] = df['set 2 total area [m2]'] / df['tot disc. area [m2]']
df['set_3_ratio'] = df['set 3 total area [m2]'] / df['tot disc. area [m2]']
df['rand_set_ratio'] = df['random set total area [m2]'] / df['tot disc. area [m2]']

# compute number of discontinuities
df['n_discs'] = params.compute_n_disc_sets(df)
# compute number of discontinuities and the corresponding joint number from the
# Q-system
df['Qsys_Jn'] = params.compute_Jn(df)
df['Q_struct'] = df['avg. RQD'] / df['Qsys_Jn']

# compute average discontinuity spacing
df['avg. disc. set spacing [m]'] = df[['meas. spacing set 1 [m]',
                                       'meas. spacing set 2 [m]',
                                       'meas. spacing set 3 [m]']].mean(axis=1)

df['Jv Sonmez & Ulusay (2002)'] = params.Jv_Sonmez2002(
    df['n_discs'], df['avg. disc. set spacing [m]'])


# compute angles between discontinuity sets where they are planar
n_vecs_1 = m.normal_vectors(df['set 1 - dip [°]'],
                            df['set 1 - dip direction [°]'])
n_vecs_2 = m.normal_vectors(df['set 2 - dip [°]'],
                            df['set 2 - dip direction [°]'])
n_vecs_3 = m.normal_vectors(df['set 3 - dip [°]'],
                            df['set 3 - dip direction [°]'])
df['alpha [°]'] = m.angle_between(n_vecs_1, n_vecs_2)
df['beta [°]'] = m.angle_between(n_vecs_1, n_vecs_3)
df['gamma [°]'] = m.angle_between(n_vecs_2, n_vecs_3)

# block volume acc. to Palmstrøm
df['block volume computed [m³]'] = params.block_volume_palmstroem(
    S1=df['meas. spacing set 1 [m]'],
    S2=df['meas. spacing set 2 [m]'],
    S3=df['meas. spacing set 3 [m]'],
    alpha=df['alpha [°]'],
    beta=df['beta [°]'],
    gamma=df['gamma [°]'])
print('standard rock mass characterization parameters computed\n')

##########################################
# compute advanced raster analyses parameters
##########################################

# add new empty columns
if DO_BLOCK_ANALYSES is True:
    new_cols = COMPLEXITY_COLUMNS + BLOCK_COLUMNS
else:
    new_cols = COMPLEXITY_COLUMNS
for col in new_cols:
    df[col] = np.nan

# compute fractal dimensions
print('computing fractal dimensions')
cols_disc_voxels = [c for c in df.columns if 'n disc. voxels' in c]
for sample in tqdm(df.index):
    # only compute for samples with all boxes calculated
    if df.loc[sample, cols_disc_voxels].isna().sum() == 0:
        df.loc[sample, 'Minkowski dimension'] = params.Minkowski(
            df.loc[sample, cols_disc_voxels].values.astype(int),
            np.array(RASTER_RESOLUTIONS))
n_processed = len(df) - df['Minkowski dimension'].isna().sum()
print(f'{n_processed} / {len(df)} fractal dimensions computed')

# compute Shannon entropy
df['Shannon entropy'] = params.Shannon_Entropy(df)
print('Shannon entropy computed\n')

# compute complexity and block metrics
print('computing complexity metrices and blocks')
resolution = RASTER_RESOLUTIONS[-1]  # -1 = finest resolution
for sample in tqdm(df.index):
    fp = fr'..\rasters\{sample}_{resolution}.pkl.gz'
    try:
        with gzip.open(fp, 'rb') as f:
            decompressed_voxel_array = pickle.load(f)

        # compute Euler Charcteristic
        euler_characteristic1 = euler_number(decompressed_voxel_array,
                                             connectivity=1)
        df.loc[sample, 'Euler characteristic'] = euler_characteristic1
        euler_characteristic2 = euler_number(1-decompressed_voxel_array,
                                             connectivity=1)
        df.loc[sample, 'Euler characteristic inverted'] = euler_characteristic2

        # structural complexity acc. to Bagrov et al. (2020)
        c = params.structural_complexity(decompressed_voxel_array,
                                         mode='3Dgrid')
        df.loc[sample, 'structural complexity'] = c

        # compression complexity
        compression_ratio = params.compression_complexity(
            decompressed_voxel_array)
        df.loc[sample, 'compression ratio'] = compression_ratio

        if DO_BLOCK_ANALYSES is True:  # experimental
            # TODO implement block sieve curve and DXX parameters
            block_array, num_blocks = utils.identify_intact_rock_regions(
                decompressed_voxel_array)
            df.loc[sample, 'n blocks'] = num_blocks - 1  # no discontinuities

            block_ids, voxels_per_block = np.unique(block_array,
                                                    return_counts=True)
            m3_per_block = voxels_per_block[1:] * (resolution**3)
            df.loc[sample, 'avg. block volume [m3]'] = m3_per_block.mean()
            df.loc[sample, 'median block volume [m3]'] = np.median(m3_per_block)
            # TODO implement block shape analyses

            if SAVE_BLOCKS is True:
                utils.array_to_pointcloud(
                    block_array, resolution=resolution,
                    savepath=fr'../output/{sample}_blocks.zip')
    except FileNotFoundError:
        pass

n_processed = len(df) - df['structural complexity'].isna().sum()
print(f'{n_processed} / {len(df)} complexities computed\n')

##########################################
# save data to excel files
##########################################

df.to_excel(r'../output/PDD1_2.xlsx')
print('data saved!')
