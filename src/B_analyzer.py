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
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from X_library import plotter, math, parameters


def fit_trend_poly(x_data, y_data, order=1):
    z_d = np.polyfit(x_data, y_data, order)
    p_d = np.poly1d(z_d)
    p_d_coeffs = p_d.c
    r2_d = r2_score(y_data, p_d(x_data))
    return p_d, r2_d, p_d_coeffs


def min_max_scaler(x):
    x = x-x.min()
    x = x/x.max()
    return x


def assess_fit(df, x, y, dropna=False):
    df_1 = df[[x, y]]
    if dropna is True:
        df_1.dropna(inplace=True)
    df_1['x_new'] = min_max_scaler(df_1[x])
    df_1['y_new'] = min_max_scaler(df_1[y])
    if dropna is True:
        df_1.dropna(inplace=True)
    if len(df_1) < 100:
        score = 2
    else:
        score = r2_score(df_1['x_new'], df_1['y_new'])
    return score
    # print(f'{x} vs. {y}: R2 {round(score, 3)}')


##########################################
# compute dirreferent additional parameters
##########################################

pltr = plotter()
m = math()
params = parameters()

pd.options.mode.chained_assignment = None


df = pd.read_excel(r'../output/dataset.xlsx', index_col='identifier')

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
                               'set 3 theo area [m2]', 'random set - n joints']].sum(axis=1)

df['set_1_ratio'] = df['set 1 theo area [m2]'] / df['tot theo area [m2]']
df['set_2_ratio'] = df['set 2 theo area [m2]'] / df['tot theo area [m2]']
df['set_3_ratio'] = df['set 3 theo area [m2]'] / df['tot theo area [m2]']
df['rand_set_ratio'] = df['set rand theo area [m2]'] / df['tot theo area [m2]']

# print(df[['set_1_ratio', 'set_2_ratio', 'set_3_ratio', 'rand_set_ratio']])

df['total joints'] = df[['set 1 - n joints', 'set 2 - n joints',
                         'set 3 - n joints', 'random set - n joints']].sum(axis=1)

df['avg. RQD'] = np.mean(df[['RQD Y', 'RQD X', 'RQD Z']].values, axis=1)
df['Qsys_Jn'] = params.Qsys_Jn(df['set_1_ratio'].values, df['set_2_ratio'].values,
                               df['set_3_ratio'].values, df['rand_set_ratio'].values,
                               df['total joints'].values)
df['Q_struct'] = df['avg. RQD'] / df['Qsys_Jn']

# TODO make weighted average
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
n_vecs_1 = m.normal_vectors(df['set 1 - dip [°]'], df['set 1 - dip direction [°]'])
n_vecs_2 = m.normal_vectors(df['set 2 - dip [°]'], df['set 2 - dip direction [°]'])
n_vecs_3 = m.normal_vectors(df['set 3 - dip [°]'], df['set 3 - dip direction [°]'])
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

print(gkl)
##########################################
# analysis
##########################################

# fitting a function to Jv_measured vs. structural complexity 
# from scipy.special import gamma
# from scipy.optimize import curve_fit

# def cust_gamma(x, alpha, beta, a):
#     # based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
#     return (beta**alpha * x**(alpha - 1) * np.exp(-beta*x))/(gamma(alpha)) * a

# df = df[df['Jv measured [discs/m³]'] < 100]

# popt, pcov = curve_fit(cust_gamma, df['Jv measured [discs/m³]'].values,
#                         df['structural complexity'].values)

# df.sort_values('Jv measured [discs/m³]', inplace=True)
# df.dropna(subset='structural complexity', inplace=True)

df.dropna(subset='structural complexity', inplace=True)
print(len(df))
base_features = ['set 1 - radius [m]', 'set 1 - radius std [m]',
                 'set 2 - radius [m]', 'set 2 - radius std [m]',
                 'set 3 - radius [m]', 'set 3 - radius std [m]',
                 'random set - radius [m]', 'random set - radius std [m]',
                 'meas. spacing set 1 [m]', 'meas. spacing set 2 [m]',
                 'meas. spacing set 3 [m]', #'RQD Y', 'RQD X', 'RQD Z',
                 # 'apparent spacing Y [m]', 'apparent spacing X [m]',
                 # 'apparent spacing Z [m]', 'P10 Y', 'P10 X', 'P10 Z', 'P20 Y',
                 # 'P21 Y', 'P20 X', 'P21 X', 'P20 Z', 'P21 Z',
                 'avg. RQD', 'avg. P10', 'avg. P20', 'avg. P21',
                 'Qsys_Jn', 'avg_angle'  # 'alpha', 'beta', 'gamma'
                 ]

from feature_engineering import feature_engineer

fe = feature_engineer()
df = fe.make_first_level_features(df, features=base_features,
                                  operations=['log', 'sqrt', 'sqr', '1div'])
l1_features = [f for f in df.columns if '-l1' in f]
print('level 1 features computed', len(df.columns))

df = fe.make_second_level_features(df, features=base_features + l1_features)
# drop features that are all 0 or have many NaN
id_0 = np.where(df.sum(axis=0).values == 0)[0]
df.drop(columns=df.columns[id_0], inplace=True)
id_nan = np.where(df.isna().sum().values > 100)[0]
df.drop(columns=df.columns[id_nan], inplace=True)
l2_features = [f for f in df.columns if '-l2' in f]
print('level 2 features computed', len(df.columns))

df = fe.make_third_level_features(df, features=base_features + l1_features)
l3_features = [f for f in df.columns if '-l3' in f]
print('level 3 features computed', len(df.columns))

scores_struct = []
scores_Jv = []
# best so far:
# sqrt_avg. P10-l1_times_sqrt_avg. RQD-l1_times_avg. RQD-l3 0.8712687640688137
# log_avg. P20-l1_plus_avg. P21_plus_avg. P20-l3 0.9946053263514517

all_features = base_features + l1_features + l2_features + l3_features
n_all_features = len(all_features)
for i, f in enumerate(all_features):
    if i % 10_000 == 0:
        print(f'{i} of {n_all_features} done')
    scores_struct.append(assess_fit(df, x='structural complexity', y=f, dropna=True))
    scores_Jv.append(assess_fit(df, x='Jv measured [discs/m³]', y=f, dropna=True))

scores_struct, scores_Jv, all_features = np.array(scores_struct), np.array(scores_Jv), np.array(all_features)

for scores in [scores_struct, scores_Jv]:
    id_fails = np.where(scores == 2)[0]
    scores = np.delete(scores, id_fails)
    all_features_new = np.delete(all_features, id_fails)

    feature_max_score = all_features_new[np.argmax(scores)]
    print(feature_max_score, max(scores))
    sorted_features = np.array(all_features_new)[np.argsort(scores)]
    scores = np.sort(scores)

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# ax1.scatter(df['structural complexity'], df[feature_max_score])
# ax1.set_xlabel('structural complexity')
# ax1.set_ylabel(feature_max_score)
# ax2.scatter(df['Jv measured [discs/m³]'], df[feature_max_score])
# ax2.set_xlabel('Jv measured [discs/m³]')
# ax2.set_ylabel(feature_max_score)
print(ghjkl)

##########################################
# plotting
##########################################

fig = plt.figure(figsize=(7.87, 7.87))

ax = fig.add_subplot(3, 1, 1)
ax.scatter(df['Jv measured [discs/m³]'], df['structural complexity'],
           edgecolor='black', color='grey', alpha=0.5)
# for i in range(len(df)):
#     x, y = df['Jv measured [discs/m³]'].iloc[i], df['structural complexity'].iloc[i]
#     ax.text(x, y, s=df.index[i])

ax.set_ylabel('structural complexity')
ax.grid(alpha=0.5)

ax = fig.add_subplot(3, 1, 2)
ax.scatter(df['Jv measured [discs/m³]'], df['avg. RQD'],
           edgecolor='black', color='grey', alpha=0.5)
ax.set_ylabel('avg. RQD')
ax.grid(alpha=0.5)

ax = fig.add_subplot(3, 1, 3)
ax.scatter(df['Jv measured [discs/m³]'], df['Minkowski'],
           edgecolor='black', color='grey', alpha=0.5)
ax.set_ylabel('Minkowski')
ax.grid(alpha=0.5)

ax.set_xlabel('Jv measured [discs/m³]')

plt.tight_layout()
plt.savefig(r'../output/data.png', dpi=300)
plt.close()


for file in os.listdir('graphics'):
    os.remove(fr'../graphics/{file}')

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(1,1,1)

for i, jv in enumerate(['Jv ISO 14689 [discs/m³]',
                        'Jv Palmstrøm 2005 [discs/m³]',
                        'Jv Sonmez & Ulusay (1999) 1',
                        'Jv Sonmez & Ulusay (1999) 2']):
    # ax = fig.add_subplot(2, 2, i+1)
    x, y = 'Jv measured [discs/m³]', jv
    r2 = r2_score(df[x], df[y])
    # ax.text(0.05, 0.95, f'R2: {round(r2, 3)}', horizontalalignment='left',
    #         verticalalignment='center', transform=ax.transAxes)
    ax.scatter(df[x], df[y], alpha=0.5, label=f'{jv}; R2: {round(r2, 2)}')

max_ = 100  # df[[x, y]].values.flatten().max()
ax.set_xlim(left=0, right=max_)
ax.set_ylim(bottom=0, top=max_)
ax.grid(alpha=0.5)
ax.set_xlabel(x)
ax.set_ylabel('Jv computed')
ax.plot([0, max_], [0, max_], color='black')
ax.legend()
plt.tight_layout()
plt.savefig(r'../graphics/_JVs.png')
plt.close()


# fig, ax = plt.subplots()
# ax.bar(x=np.arange(len(corr_params)), height=correlations, tick_label=corr_params)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
# ax.set_ylabel('correlation coefficient with\nHausdorff dimension')
# ax.grid(alpha=0.5)
# plt.tight_layout()


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
        plt.savefig(fr'graphics\{params_dict[x]}_{params_dict[y]}.png', dpi=150)
        plt.close()
