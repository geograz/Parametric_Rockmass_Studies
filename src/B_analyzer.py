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

df = pd.read_excel(r'../output/dataset.xlsx', index_col='identifier')

##########################################
# compute dirreferent additional parameters
##########################################

df['avg. P10'] = np.mean(df[['P10 Y', 'P10 X', 'P10 Z']].values, axis=1)
df['avg. P20'] = np.mean(df[['P20 Y', 'P20 X', 'P20 Z']].values, axis=1)
df['avg. P21'] = np.mean(df[['P21 Y', 'P21 X', 'P21 Z']].values, axis=1)

df['m_length'] = np.full(len(df), 10)
df['n_intersections'] = df['avg. P10'] * df['m_length']

# compute different versions of the volumetric joint count acc. to literature
df['Jv ISO 14689 [discs/m³]'] = params.Jv_ISO14689(
    df['meas. spacing set 1 [m]'], df['meas. spacing set 2 [m]'],
    df['meas. spacing set 3 [m]'])

df['Jv Palmstrøm 2005 [discs/m³]'] = params.Jv_Palmstroem2005(
    df['meas. spacing set 1 [m]'], df['meas. spacing set 2 [m]'],
    df['meas. spacing set 3 [m]'], df['random set - n joints'],
    df['bounding box size [m]']**3)

df['Jv Sonmez & Ulusay (1999) 1'] = params.Jv_Sonmez1999_1(df['P10 X'],
                                                           df['P10 Y'],
                                                           df['P10 Z'])

df['Jv Sonmez & Ulusay (1999) 2'] = params.Jv_Sonmez1999_2(df['P10 X'])

# compute set ratio based on theo disc. area per set
df['set 1 theo area [m2]'] = (df['set 1 - radius [m]']**2 * np.pi) * df['set 1 - n joints']
df['set 2 theo area [m2]'] = (df['set 2 - radius [m]']**2 * np.pi) * df['set 2 - n joints']
df['set 3 theo area [m2]'] = (df['set 3 - radius [m]']**2 * np.pi) * df['set 3 - n joints']
df['set rand theo area [m2]'] = (df['random set - radius [m]']**2 * np.pi) * df['random set - n joints']
df['tot theo area [m2]'] = df[['set 1 theo area [m2]', 'set 2 theo area [m2]',
                               'set 3 theo area [m2]',
                               'random set - n joints']].sum(axis=1)

df['set_1_ratio'] = df['set 1 theo area [m2]'] / df['tot theo area [m2]']
df['set_2_ratio'] = df['set 2 theo area [m2]'] / df['tot theo area [m2]']
df['set_3_ratio'] = df['set 3 theo area [m2]'] / df['tot theo area [m2]']
df['rand_set_ratio'] = df['set rand theo area [m2]'] / df['tot theo area [m2]']

df['total joints'] = df[['set 1 - n joints', 'set 2 - n joints',
                         'set 3 - n joints',
                         'random set - n joints']].sum(axis=1)

df['avg. RQD'] = np.mean(df[['RQD Y', 'RQD X', 'RQD Z']].values, axis=1)
df['n_discs'], df['Qsys_Jn'] = params.compute_n_disc_sets(
    df['set_1_ratio'].values,
    df['set_2_ratio'].values,
    df['set_3_ratio'].values,
    df['rand_set_ratio'].values,
    df['total joints'].values)
df['Q_struct'] = df['avg. RQD'] / df['Qsys_Jn']

df['avg. radius [m]'] = (df['set 1 - radius [m]'].fillna(0)
                         + df['set 2 - radius [m]'].fillna(0)
                         + df['set 3 - radius [m]'].fillna(0)
                         + df['random set - radius [m]'].fillna(0)) / 4

df['avg. app. spacing [m]'] = np.mean(df[['apparent spacing Y [m]',
                                          'apparent spacing X [m]',
                                          'apparent spacing Z [m]']].values,
                                      axis=1)

df['block aspect ratio'] = df['a3'] / df['a1']

# compute angles between joint sets
n_vecs_1 = m.normal_vectors(df['set 1 - dip [°]'],
                            df['set 1 - dip direction [°]'])
n_vecs_2 = m.normal_vectors(df['set 2 - dip [°]'],
                            df['set 2 - dip direction [°]'])
n_vecs_3 = m.normal_vectors(df['set 3 - dip [°]'],
                            df['set 3 - dip direction [°]'])
df['alpha'] = m.angle_between(n_vecs_1, n_vecs_2)
df['beta'] = m.angle_between(n_vecs_1, n_vecs_3)
df['gamma'] = m.angle_between(n_vecs_2, n_vecs_3)
df['avg_angle'] = np.mean(df[['alpha', 'beta', 'gamma']].values, axis=1)
df['min_angle'] = np.min(df[['alpha', 'beta', 'gamma']].values, axis=1)
df['max_angle'] = np.max(df[['alpha', 'beta', 'gamma']].values, axis=1)

df['std_dipdir'] = np.mean(df[['set 1 - dip direction std [°]',
                               'set 2 - dip direction std [°]',
                               'set 3 - dip direction std [°]',
                               ]].values, axis=1)
df['std_dip'] = np.mean(df[['set 1 - dip std [°]', 'set 2 - dip std [°]',
                            'set 3 - dip std [°]']].values, axis=1)

# block volume acc. to Palmstrøm
df['block volume computed'] = params.block_volume_palmstroem(
    S1=df['meas. spacing set 1 [m]'],
    S2=df['meas. spacing set 2 [m]'],
    S3=df['meas. spacing set 3 [m]'],
    alpha=df['alpha'],
    beta=df['beta'],
    gamma=df['gamma'])

# save data to excel file
df.to_excel(r'../output/dataset1.xlsx')

##########################################
# plotting
##########################################

# 0.9535247683525085 - with only m_length and n_intersections
df['StructC_Erharter'] = df['n_intersections'] / (np.sqrt(df['n_intersections'])**np.log(df['n_intersections'])) / (1 / df['n_intersections'] ** np.sqrt(df['m_length']))
# 0.9892848134040833 - with only m_length and n_intersections
df['Jv_Erharter'] = np.sqrt(df['m_length']) + np.log(df['n_intersections']) / np.log(df['m_length']) + (df['n_intersections'] / 10) * np.log(df['m_length'])
# 0.9944955706596375 - with only m_length and n_intersections
df['Mink_Erharter'] = df['n_intersections'] / ((df['m_length']**2) + df['n_intersections']) / (np.log(df['n_intersections']) + np.log(df['m_length']))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['Jv measured [discs/m³]'], df['structural complexity'],
           color='grey', edgecolor='black', alpha=0.5)
ax.grid(alpha=0.5)

ax2 = ax.twinx()
ax2.scatter(df['Jv measured [discs/m³]'], df['StructC_Erharter'],
            color='orange', edgecolor='black', alpha=0.5)

ax.set_xlabel('Jv measured [discs/m³]')
ax.set_ylabel('structural complexity')
ax2.set_ylabel('structural complexity computed')

plt.tight_layout()
plt.savefig(r'../graphics/structs.png', dpi=400)
plt.close()


fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['Jv measured [discs/m³]'], df['Minkowski'],
           color='grey', edgecolor='black', alpha=0.5)
ax.grid(alpha=0.5)

ax2 = ax.twinx()
ax2.scatter(df['Jv measured [discs/m³]'], df['Mink_Erharter'],
            color='orange', edgecolor='black', alpha=0.5)

ax.set_xlabel('Jv measured [discs/m³]')
ax.set_ylabel('Minkowski dimension')
ax2.set_ylabel('Minkowski dimension computed')

plt.tight_layout()
plt.savefig(r'../graphics/minks.png', dpi=400)
plt.close()


pltr.DEM_FEM_data(df)

pltr.Jv_plot(df, Jv_s=['Jv ISO 14689 [discs/m³]',
                       'Jv Palmstrøm 2005 [discs/m³]',
                       'Jv Sonmez & Ulusay (1999) 1',
                       'Jv Sonmez & Ulusay (1999) 2', 'Jv_Erharter'])

for file in os.listdir(r'../graphics/scatters/'):
    os.remove(fr'../graphics/scatters/{file}')

plot_params = ['structural complexity', 'Minkowski', 'Jv measured [discs/m³]',
               'avg. RQD', 'P32', 'avg. P10', 'avg. P20', 'avg. P21',
               'avg. app. spacing [m]', 'max block volume [m³]',
               'avg. block volume [m³]', 'avg. block edge length [m]',
               'avg. block surface area [m²]', 'n blocks', 'a3', 'a2', 'a1',
               'block aspect ratio', 'Q_struct', 'avg_angle',
               'min_angle', 'max_angle', 'std_dipdir', 'std_dip',
               'block volume computed', 'similarity n zeros', 'similarity max',
               'similarity min', 'similarity mean', 'similarity median',
               'total joints']

pltr.scatter_combinations(df, plot_params)
