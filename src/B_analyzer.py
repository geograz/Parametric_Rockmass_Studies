# -*- coding: utf-8 -*-
"""
Code to the paper "Rock mass structure characterization considering finite and
folded discontinuities"
Dr. Georg H. Erharter - 2023
DOI: https://doi.org/10.1007/s00603-024-03787-9

Script that processes the compiled records of the discrete discontinuity
dataset, computes new parameters and creates figures to visualize the dataset.
"""

import gzip
import numpy as np
import pandas as pd
import pickle
from scipy.stats import entropy
from tqdm import tqdm
import zlib

from X_library import plotter, math, parameters


##########################################
# static variables and constants
##########################################

RASTER_RESOLUTIONS = [0.25, 0.2, 0.15, 0.1, 0.05]

PLOT_PARAMS = ['avg. P10', 'avg. P20', 'avg. P21', 'P32',
               'Jv measured [discs/m³]', 'avg. RQD', 'Minkowski dimension']

RELATION_DIC = {'linear':
                [['avg. P10', 'avg. P21'], ['avg. P21', 'avg. P10'],
                 ['avg. P21', 'P32'], ['P32', 'avg. P21'],
                 ['avg. P10', 'P32'], ['P32', 'avg. P10'],
                 ['avg. P10', 'Jv measured [discs/m³]'], ['Jv measured [discs/m³]', 'avg. P10'],
                 ['avg. P21', 'Jv measured [discs/m³]'], ['Jv measured [discs/m³]', 'avg. P21'],
                 ['P32', 'Jv measured [discs/m³]'], ['Jv measured [discs/m³]', 'P32']],

                'exponential':
                [['avg. RQD', 'avg. P10'], ['avg. P10', 'avg. RQD'],
                 ['avg. P20', 'avg. RQD'], ['avg. RQD', 'avg. P20'],
                 ['avg. P21', 'avg. RQD'], ['avg. RQD', 'avg. P21'],
                 ['Jv measured [discs/m³]', 'avg. RQD'], ['avg. RQD', 'Jv measured [discs/m³]'],
                 ['P32', 'avg. RQD'], ['avg. RQD', 'P32']],

                'powerlaw':
                [['Minkowski dimension', 'avg. P21'], ['avg. P21', 'Minkowski dimension'],
                 ['Minkowski dimension', 'avg. P10'], ['avg. P10', 'Minkowski dimension'],
                 ['Minkowski dimension', 'avg. P20'], ['avg. P20', 'Minkowski dimension'],
                 ['Minkowski dimension', 'P32'], ['P32', 'Minkowski dimension'],
                 ['Minkowski dimension', 'Jv measured [discs/m³]'], ['Jv measured [discs/m³]', 'Minkowski dimension'],
                 ['avg. P20', 'avg. P10'], ['avg. P10', 'avg. P20'],
                 ['avg. P20', 'avg. P21'], ['avg. P21', 'avg. P20'],
                 ['avg. P20', 'P32'], ['P32', 'avg. P20'],
                 ['avg. P20', 'Jv measured [discs/m³]'], ['Jv measured [discs/m³]', 'avg. P20'],
                 ['Minkowski dimension', 'avg. RQD'], ['avg. RQD', 'Minkowski dimension']],
                }

##########################################
# instantiations, data loading and preprocessing
##########################################

pltr = plotter()
m = math()
params = parameters()

pd.options.mode.chained_assignment = None

df = pd.read_excel(r'../output/PDD1_0.xlsx', index_col='identifier')

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
cols_other = ['Minkowski dimension', 'structural complexity',
              'compression ratio']
for col in cols_other:
    df[col] = np.nan

# compute fractal dimensions
print('computing fractal dimensions')
cols_disc_voxels = [c for c in df.columns if 'n disc. voxels' in c]
for sample in tqdm(df.index):
    if df.loc[sample, cols_disc_voxels].isna().sum() == 0:
        df.loc[sample, 'Minkowski dimension'] = params.Minkowski(
            df.loc[sample, cols_disc_voxels].values.astype(int),
            np.array(RASTER_RESOLUTIONS))

n_processed = len(df) - df['Minkowski dimension'].isna().sum()
print(f'{n_processed} / {len(df)} fractal dimensions computed')

# compute Shannon entropy
Shannon_cols = ['n empty voxels at 0.05 [m]', 'n disc. voxels at 0.05 [m]']
counts = df[Shannon_cols].values
df['Shannon entropy'] = entropy(counts, base=2, axis=1)
print('Shannon entropy computed\n')

# compute complexity metrics
print('computing complexity metrices')
for sample in tqdm(df.index):
    fp = fr'..\rasters\{sample}_0.05.pkl.gz'
    try:
        with gzip.open(fp, 'rb') as f:
            decompressed_voxel_array = pickle.load(f)
        # structural complexity acc. to Bagrov et al. (2020)
        c = params.structural_complexity(decompressed_voxel_array,
                                         mode='3Dgrid')
        df.loc[sample, 'structural complexity'] = c
        # compression complexity
        flattened = decompressed_voxel_array.flatten()
        byte_data = flattened.tobytes()  # Convert to bytes for compression
        compressed_data = zlib.compress(byte_data)  # Compress the byte data
        compression_ratio = len(compressed_data) / len(byte_data)
        df.loc[sample, 'compression ratio'] = compression_ratio
    except FileNotFoundError:
        pass

n_processed = len(df) - df['structural complexity'].isna().sum()
print(f'{n_processed} / {len(df)} complexities computed\n')

##########################################
# save data to excel files
##########################################

df.to_excel(r'../output/PDD1_1.xlsx')
print('data saved -> plotting\n')

##########################################
# visualizations of the dataset
##########################################

pltr.struct_complex_plot(df, close=True)

pltr.custom_pairplot(df, PLOT_PARAMS, RELATION_DIC)

pltr.scatter_combinations(df, RELATION_DIC, PLOT_PARAMS)

pltr.Q_Jv_plot(df)

pltr.directional_lineplot(df)

pltr.Pij_plot(df)

pltr.RQD_spacing_hist_plot(df)

JVs = ['Jv ISO 14689 (2019)', 'Jv Palmstrøm (2000)',
       'Jv Sonmez & Ulusay (1999) 1', 'Jv Sonmez & Ulusay (1999) 2',
       'Jv Sonmez & Ulusay (2002)', 'Jv Erharter (2023)']
pltr.Jv_plot(df, Jv_s=JVs,
             limit=df['Jv measured [discs/m³]'].max()+5)
