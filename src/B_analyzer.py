# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:38:57 2023

@author: GEr
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from X_library import plotter, math, parameters, utilities


pltr = plotter()
m = math()
params = parameters()
utils = utilities()

pd.options.mode.chained_assignment = None

df = pd.read_excel(r'../output/PDD1_0.xlsx', index_col='identifier')

id_JV_max = df['Jv measured [discs/m³]'].argmax()
id_JV_min = df['Jv measured [discs/m³]'].argmin()

print(df['Jv measured [discs/m³]'].describe())
print(f'sample with lowest JV: {df.index[id_JV_min]}')
print(f'sample with highest JV: {df.index[id_JV_max]}')

##########################################
# compute dirreferent additional parameters
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

df['Jv Erharter (2023)'] = 3*df['avg. P10'] * 0.75 + 4

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

measured = ['meas. spacing set 1 [m]', 'meas. spacing set 2 [m]',
            'meas. spacing set 3 [m]', 'RQD Y', 'RQD X', 'RQD Z',
            'apparent spacing Y [m]', 'apparent spacing X [m]',
            'apparent spacing Z [m]', 'P10 Y', 'P10 X', 'P10 Z', 'P20 X',
            'P21 X', 'P20 Y', 'P21 Y', 'P20 Z', 'P21 Z',
            'Jv measured [discs/m³]', 'P32', 'set 1 total area [m2]',
            'set 2 total area [m2]', 'set 3 total area [m2]',
            'random set total area [m2]', 'Minkowski dimension']
computed = ['avg. P10', 'avg. P20', 'avg. P21', 'avg. app. spacing [m]',
            'avg. RQD', 'Jv ISO 14689 (2019)', 'Jv Palmstrøm (2000)',
            'Jv Sonmez & Ulusay (1999) 1', 'Jv Sonmez & Ulusay (1999) 2',
            'Jv Erharter (2023)', 'tot disc. area [m2]', 'set_1_ratio',
            'set_2_ratio', 'set_3_ratio', 'rand_set_ratio', 'n_discs',
            'Qsys_Jn', 'Q_struct', 'avg. disc. set spacing [m]',
            'Jv Sonmez & Ulusay (2002)', 'alpha [°]', 'beta [°]', 'gamma [°]',
            'block volume computed [m³]']
df[measured].describe().to_excel(r'../output/PDD1_stats_measured.xlsx')
df[computed].describe().to_excel(r'../output/PDD1_stats_computed.xlsx')

# save data to excel file
df.to_excel(r'../output/PDD1_1.xlsx')

##########################################
# visualizations of the dataset
##########################################

pltr.Q_Jv_plot(df)

pltr.directional_lineplot(df)

pltr.Pij_plot(df)

pltr.RQD_spacing_hist_plot(df)

# pltr.DEM_FEM_data(df)

pltr.Jv_plot(df, Jv_s=['Jv ISO 14689 (2019)',
                       'Jv Palmstrøm (2000)',
                       'Jv Sonmez & Ulusay (1999) 1',
                       'Jv Sonmez & Ulusay (1999) 2',
                       'Jv Sonmez & Ulusay (2002)',
                       'Jv Erharter (2023)'],
             limit=df['Jv measured [discs/m³]'].max()+5)

for file in os.listdir(r'../graphics/scatters/'):
    os.remove(fr'../graphics/scatters/{file}')

plot_params = ['avg. RQD', 'avg. P10', 'avg. P20', 'avg. P21', 'P32',
               'avg. app. spacing [m]', 'Jv measured [discs/m³]',
               'block volume computed [m³]', 'Minkowski dimension'
               ]

pltr.scatter_combinations(df, plot_params)
