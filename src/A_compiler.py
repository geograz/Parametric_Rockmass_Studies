# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:11:14 2022

@author: GEr
"""

import numpy as np
from os import listdir, remove
import pandas as pd

from X_library import parameters


params = parameters()

pd.options.mode.chained_assignment = None

##########################################
# load text files with geometry parameters as outputted by grasshopper

# clean files in case grasshopper produced inclomplete files
all_files = list(listdir(r'../combinations'))
for file_name in all_files:
    id_ = file_name[:12]
    values = f'{id_}.txt'
    mesh = f'{id_}_discontinuities.stl'
    boxes = f'{id_}_boxcount.txt'
    if values in all_files and mesh in all_files:
        pass
    else:
        print('delete incomplete files')
        for f in [fr'../combinations/{values}', fr'../combinations/{mesh}',
                  fr'../combinations/{boxes}']:
            try:
                remove(f)
            except FileNotFoundError:
                pass

# load data from text files to a pd dataframe
contents = []
for file_name in listdir(r'../combinations'):
    if '_box' in file_name or 'FAIL' in file_name or '_Rast' in file_name:
        pass
    else:
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

inputs = ['bounding box size [m]', 'Jv boxes edge size [m]', 'seed',
          'set 1 - n joints',
          'set 1 - radius [m]', 'set 1 - radius std [m]',
          'set 1 - dip direction [°]', 'set 1 - dip direction std [°]',
          'set 1 - dip [°]', 'set 1 - dip std [°]', 'set 1 - type',
          'F_rand_sin', 'F_rand_n_planes', 'F_rand_angle', 'F_rand_axis_x',
          'F_rand_axis_y', 'F_rand_axis_z',
          'set 2 - n joints', 'set 2 - radius [m]', 'set 2 - radius std [m]',
          'set 2 - dip direction [°]', 'set 2 - dip direction std [°]',
          'set 2 - dip [°]', 'set 2 - dip std [°]',
          'set 3 - n joints',
          'set 3 - radius [m]', 'set 3 - radius std [m]',
          'set 3 - dip direction [°]', 'set 3 - dip direction std [°]',
          'set 3 - dip [°]', 'set 3 - dip std [°]',
          'random set - n joints', 'random set - radius [m]',
          'random set - radius std [m]',
          'identifier']

outputs = ['meas. spacing set 1 [m]', 'meas. spacing set 2 [m]',
           'meas. spacing set 3 [m]', 'RQD Y', 'RQD X', 'RQD Z',
           'apparent spacing Y [m]', 'apparent spacing X [m]',
           'apparent spacing Z [m]',
           'P10 Y', 'P10 X', 'P10 Z', 'P20 X', 'P21 X', 'P20 Y', 'P21 Y',
           'P20 Z', 'P21 Z', 'Jv measured [discs/m³]', 'P32',
           'n blocks', 'avg. block volume [m³]',
           'max block volume [m³]', 'min block volume [m³]',
           'avg. block edge length [m]',
           'avg. block surface area [m²]', 'a3', 'a2', 'a1',
           'set 1 total area [m2]', 'set 2 total area [m2]',
           'set 3 total area [m2]', 'random set total area [m2]']

columns = inputs + outputs

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

print('data frame set up')
# set up empty columns for later population
for col in ['Minkowski dimension', 'Hausdorff', 'similarity n zeros', 'similarity max',
            'similarity min', 'similarity mean', 'similarity median',
            'structural complexity']:
    df[col] = np.nan

##########################################
# add fractal dimensions where they are computed

for sample in df.index:
    try:
        df_boxcount = pd.read_csv(fr'../combinations/{sample}_boxcount.txt')
        df['Minkowski dimension'].loc[sample] = params.Minkowski(
            df_boxcount['n boxes'], df_boxcount['box edge length [m]'])
        df['Hausdorff'].loc[sample] = params.Hausdorff(
            df_boxcount['n boxes'], df_boxcount['box edge length [m]'])
        if np.isnan(df['Minkowski dimension'].loc[sample]) == True:
            raise ValueError(f'{sample} has no computed boxes')
    except FileNotFoundError:
        pass

n_processed = len(df) - df['Minkowski dimension'].isna().sum()
print(f'{n_processed} / {len(df)} fractal dimensions computed')

##########################################
# add similarity and structural complexity information where it is computed

for sample in df.index:
    try:
        with open(fr'../combinations/{sample}_RasterAnalysis.txt', 'r') as f:
            content = []
            for v in f.read().split(','):
                if v == 'nan' or v == "nan":
                    content.append(np.nan)
                else:
                    content.append(eval(v))

        for i, col in enumerate(['similarity n zeros', 'similarity max',
                                 'similarity min', 'similarity mean',
                                 'similarity median',
                                 'structural complexity']):
            if col == 'structural complexity' and content[i] == 0:
                remove(fr'../combinations/{sample}_RasterAnalysis.txt')
            else:
                df[col].loc[sample] = content[i]

    except FileNotFoundError:
        pass

n_processed = len(df) - df['structural complexity'].isna().sum()
print(f'{n_processed} / {len(df)} similarities computed')

##########################################
# some basic data preprocessing

# convert object dtypes to float
for param in df.columns:
    df[param] = pd.to_numeric(df[param])

# replace joint properties for joint sets that have no joints
for joint_set in [1, 2, 3, 4]:
    if joint_set == 4:
        id_0 = np.where(df['random set - n joints'] == 0)[0]
        df['random set - radius [m]'].iloc[id_0] = np.nan
    else:
        id_0 = np.where(df[f'set {joint_set} - n joints'] == 0)[0]
        df[f'set {joint_set} - radius [m]'].iloc[id_0] = np.nan
        df[f'set {joint_set} - dip [°]'].iloc[id_0] = np.nan
        df[f'set {joint_set} - dip direction [°]'].iloc[id_0] = np.nan

# idx_neg_vol = np.where(df['avg. block volume [m³]'] < 0)[0]
# df = df.drop(index=idx_neg_vol)

# drop all columns of features that are still under development such as block
# volumes or fractal dimensions
df.drop(['n blocks', 'avg. block volume [m³]', 'max block volume [m³]',
         'min block volume [m³]', 'avg. block edge length [m]',
         'avg. block surface area [m²]', 'a3', 'a2', 'a1',
         'Hausdorff', 'similarity n zeros', 'similarity max',
         'similarity min', 'similarity mean', 'similarity median',
         'structural complexity'],
        axis=1, inplace=True)

# save to excel file
df.to_excel(r'../output/PDD1_0.xlsx')
print('data compiled')
