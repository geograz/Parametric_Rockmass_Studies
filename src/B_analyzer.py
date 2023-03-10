# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:38:57 2023

@author: GEr
"""

from itertools import combinations
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

# print(df[['set_1_ratio', 'set_2_ratio', 'set_3_ratio', 'rand_set_ratio']])

df['total joints'] = df[['set 1 - n joints', 'set 2 - n joints',
                         'set 3 - n joints',
                         'random set - n joints']].sum(axis=1)

df['avg. RQD'] = np.mean(df[['RQD Y', 'RQD X', 'RQD Z']].values, axis=1)
df['Qsys_Jn'] = params.Qsys_Jn(df['set_1_ratio'].values,
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
df['block volume computed'] = params.block_volume_palmstroem(S1=df['meas. spacing set 1 [m]'],
                                                             S2=df['meas. spacing set 2 [m]'],
                                                             S3=df['meas. spacing set 3 [m]'],
                                                             alpha=df['alpha'],
                                                             beta=df['beta'],
                                                             gamma=df['gamma'])

df.to_excel(r'../output/dataset1.xlsx')

##########################################
# analysis
##########################################

df_features = df.dropna(subset='structural complexity')
print(len(df_features))
base_features = ['set 1 - radius [m]', 'set 1 - radius std [m]',
                 'set 2 - radius [m]', 'set 2 - radius std [m]',
                 'set 3 - radius [m]', 'set 3 - radius std [m]',
                 'random set - radius [m]', 'random set - radius std [m]',
                 'meas. spacing set 1 [m]', 'meas. spacing set 2 [m]',
                 'meas. spacing set 3 [m]',
                 'avg. RQD', 'avg. P10', 'Qsys_Jn', 'avg_angle']

fe = feature_engineer()
df_features = fe.make_first_level_features(df_features, features=base_features,
                                           operations=None)
l1_features = [f for f in df_features.columns if '-l1' in f]
print('level 1 features computed', len(df_features.columns))

df_features = fe.make_second_level_features(df_features, features=base_features + l1_features)
# drop features that are all 0 or have many NaN
id_0 = np.where(df_features.sum(axis=0).values == 0)[0]
df_features.drop(columns=df_features.columns[id_0], inplace=True)
id_nan = np.where(df_features.isna().sum().values > 100)[0]
df_features.drop(columns=df_features.columns[id_nan], inplace=True)
l2_features = [f for f in df_features.columns if '-l2' in f]
print('level 2 features computed', len(df_features.columns))

# df_features = fe.make_third_level_features(df_features, features=base_features + l1_features)
# l3_features = [f for f in df_features.columns if '-l3' in f]
# print('level 3 features computed', len(df_features.columns))

scores_struct = []
scores_Jv = []

all_features = base_features + l1_features + l2_features  # + l3_features
n_all_features = len(all_features)
for i, f in enumerate(all_features):
    if i % 10_000 == 0:
        print(f'{i} of {n_all_features} done')
    scores_struct.append(utils.assess_fit(df_features,
                                          x='structural complexity', y=f,
                                          dropna=True))
    scores_Jv.append(utils.assess_fit(df_features,
                                      x='Jv measured [discs/m³]', y=f,
                                      dropna=True))

for scores in [np.array(scores_struct), np.array(scores_Jv)]:
    id_fails = np.where(scores == 2)[0]
    scores = np.delete(scores, id_fails)
    all_features_new = np.delete(np.array(all_features), id_fails)

    feature_max_score = all_features_new[np.argmax(scores)]
    print(feature_max_score, max(scores))
    sorted_features = np.array(all_features_new)[np.argsort(scores)]
    scores = np.sort(scores)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax1.scatter(df_features['structural complexity'],
            df_features['sqrt_avg. P10-l1_times_sqr_avg. RQD-l1-l2'])
ax1.set_xlabel('structural complexity')
ax1.set_ylabel(feature_max_score)
ax2.scatter(df_features['Jv measured [discs/m³]'],
            df_features['sqrt_avg. P10-l1_times_sqr_avg. RQD-l1-l2'])
ax2.set_xlabel('Jv measured [discs/m³]')
ax2.set_ylabel(feature_max_score)
plt.tight_layout()
# print(ghjkl)

##########################################
# plotting
##########################################

pltr.DEM_FEM_data(df)

pltr.Jv_plot(df, Jv_s=['Jv ISO 14689 [discs/m³]',
                       'Jv Palmstrøm 2005 [discs/m³]',
                       'Jv Sonmez & Ulusay (1999) 1',
                       'Jv Sonmez & Ulusay (1999) 2'])

for file in os.listdir(r'../graphics/scatters/'):
    os.remove(fr'../graphics/scatters/{file}')

plot_params = ['structural complexity', 'Jv measured [discs/m³]',
               'Jv Sonmez & Ulusay (1999) 1','avg. RQD', 'avg. P10',
               'avg. P20', 'avg. P21', 'P32',
               'avg. app. spacing [m]', 'max block volume [m³]',
               'avg. block volume [m³]', 'avg. block edge length [m]',
               'avg. block surface area [m²]', 'n blocks', 'a3', 'a2', 'a1',
               'block aspect ratio', 'Q_struct', 'Hausdorff', 'avg_angle',
               'min_angle', 'max_angle', 'std_dipdir', 'std_dip',
               'block volume computed', 'similarity n zeros', 'similarity max',
               'similarity min', 'similarity mean', 'similarity median',
               'total joints']

params_dict = dict(zip(plot_params, list(range(len(plot_params)))))

log_scale_params = ['avg. app. spacing [m]', 'max block volume [m³]',
                    'avg. block volume [m³]', 'avg. block edge length [m]',
                    'n blocks', 'a3', 'a2', 'a1', 'block aspect ratio',
                    'avg. block surface area [m²]', 'Q_struct',
                    'block volume computed']

for x, y in list(combinations(plot_params, 2)):
    if df[x].isna().sum() == len(df) or df[y].isna().sum() == len(df):
        pass
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(df[x], df[y], alpha=0.5)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.5)
        if x in log_scale_params:
            ax.set_xscale('log')
        if y in log_scale_params:
            ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(fr'../graphics/scatters/{params_dict[x]}_{params_dict[y]}.png', dpi=150)
        plt.close()
