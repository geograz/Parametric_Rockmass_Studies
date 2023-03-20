# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:38:57 2023

@author: GEr
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from X_feature_engineering import feature_engineer
from X_library import plotter, math, parameters, utilities


pltr = plotter()
m = math()
params = parameters()
utils = utilities()
fe = feature_engineer()

pd.options.mode.chained_assignment = None

df = pd.read_excel(r'../output/dataset.xlsx', index_col='identifier')

##########################################
# compute dirreferent additional parameters
##########################################

df['avg. P10'] = np.mean(df[['P10 Y', 'P10 X', 'P10 Z']].values, axis=1)
df['avg. P20'] = np.mean(df[['P20 Y', 'P20 X', 'P20 Z']].values, axis=1)
df['avg. P21'] = np.mean(df[['P21 Y', 'P21 X', 'P21 Z']].values, axis=1)

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

df.to_excel(r'../output/dataset1.xlsx')

##########################################
# analysis
##########################################

base_features = [#'set 1 - radius [m]', 'set 2 - radius [m]',
                  #'set 3 - radius [m]', 'random set - radius [m]',
                  #'meas. spacing set 1 [m]', 'meas. spacing set 2 [m]',
                  #'meas. spacing set 3 [m]',
                  'avg. RQD', 'avg. P10', 'n_discs', 'avg. app. spacing [m]']
                  # 'avg. P20', 'avg. P21', 


df_features = df.dropna(subset=['structural complexity', 'Minkowski'])
print(len(df_features))

df_features = fe.make_1st_level_features(df_features, features=base_features,
                                         operations=None, drop_empty=True)
l1_features = [f for f in df_features.columns if '-l1' in f]

df_features = fe.make_2nd_level_features(df_features,
                                         features=base_features + l1_features,
                                         drop_empty=True)
l2_features = [f for f in df_features.columns if '-l2' in f]

all_features = base_features + l1_features + l2_features
targets = ['structural complexity', 'Jv measured [discs/m³]', 'Minkowski']
scores_struct, scores_Jv, scores_mink = utils.assess_fits(
    df_features, features=all_features, targets=targets)

struct_best, struct_max = utils.get_best_feature(scores_struct, all_features)
Jv_best, Jv_max = utils.get_best_feature(scores_Jv, all_features)
mink_best, mink_max = utils.get_best_feature(scores_mink, all_features)

print('start 3rd level feature check')
counter = 0
for l3_f, l3_f_data in fe.gen_3rd_level_features(df_features, all_features):

    l3_f_data = utils.convert_inf(l3_f_data)
    if np.isnan(l3_f_data).sum() > 0:
        # pass if data contains nan
        pass
    else:
        s_struct = utils.assess_fit2(df_features['structural complexity'].values,
                                     y=l3_f_data, scale_indiv=True)
        if s_struct > struct_max:
            struct_max = s_struct
            struct_best = l3_f
            print(f'highest struct: {struct_best} with {struct_max}')

        s_mink = utils.assess_fit2(df_features['Minkowski'].values,
                                   y=l3_f_data, scale_indiv=True)
        if s_mink > mink_max:
            mink_max = s_mink
            mink_best = l3_f
            print(f'highest minkowski: {mink_best} with {mink_max}')

        s_Jv = utils.assess_fit2(df_features['Jv measured [discs/m³]'].values,
                                 y=l3_f_data, scale_indiv=False)
        if s_Jv > Jv_max:
            Jv_max = s_Jv
            Jv_best = l3_f
            print(f'highest Jv: {Jv_best} with {Jv_max}')

    if counter % 10_000 == 0:
        print(f'{counter} 3rd level features checked')
    counter += 1

print(ghjkl)

##########################################
# plotting
##########################################

# 0.935524747650611
df['StructC_Erharter'] = (df['avg. P10'] - np.log(df['avg. app. spacing [m]'])) * (df['avg. RQD'] - 1/df['n_discs']) * (df['avg. RQD'] - 2 * df['avg. P10'])

# 0.9851848687312329
df['Jv_Erharter'] = df['avg. P10'] / 2 + df['avg. P10'] * 2 + df['n_discs']

df['Jv_Erharter'] = (df['avg. RQD'] ** (df['avg. P10'] / 10)) + df['avg. P10'] / 2 + df['avg. P10'] * 2
# 0.9840097299783439
df['Mink_Erharter'] = (df['avg. app. spacing [m]'] / 10) ** (1/df['avg. P10'])


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

plot_params = ['structural complexity', 'Jv measured [discs/m³]', 'avg. RQD',
               'avg. P10', 'avg. P20', 'avg. P21', 'P32',
               'avg. app. spacing [m]', 'max block volume [m³]',
               'avg. block volume [m³]', 'avg. block edge length [m]',
               'avg. block surface area [m²]', 'n blocks', 'a3', 'a2', 'a1',
               'block aspect ratio', 'Q_struct', 'Minkowski', 'avg_angle',
               'min_angle', 'max_angle', 'std_dipdir', 'std_dip',
               'block volume computed', 'similarity n zeros', 'similarity max',
               'similarity min', 'similarity mean', 'similarity median',
               'total joints']

pltr.scatter_combinations(df, plot_params)
