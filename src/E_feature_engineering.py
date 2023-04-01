# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:22:39 2023

@author: GEr
"""

import numpy as np
import pandas as pd
from os import listdir

from X_library import utilities
from X_feature_engineering_library import feature_engineer


fe = feature_engineer()
utils = utilities()


df = pd.read_excel(r'../output/dataset1.xlsx')


base_features = [  # 'set 1 - radius [m]', 'set 2 - radius [m]',
                 # 'set 3 - radius [m]', 'random set - radius [m]',
                 # 'meas. spacing set 1 [m]', 'meas. spacing set 2 [m]',
                 # 'meas. spacing set 3 [m]',
                 'm_length', 'n_intersections',
                 # 'n_discs',
                 # 'avg. RQD',
                 # 'avg. P10', 'avg. app. spacing [m]',
                 # 'avg. P20', 'avg. P21',
                 ]

df = df.dropna(subset=['structural complexity', 'Minkowski'])
print(len(df))

df = fe.make_1st_level_features(df, features=base_features,
                                operations=['log', 'sqrt', 'sqr',
                                            'power_3', 'mult10', 'div10',
                                            '1div'], drop_empty=True)
l1_features = [f for f in df.columns if '-l1' in f]

df = fe.make_2nd_level_features(df, features=base_features + l1_features,
                                drop_empty=True)
l2_features = [f for f in df.columns if '-l2' in f]

all_features = base_features + l1_features + l2_features
print(len(all_features))

# fe.gen_3rd_level_structure(all_features, list(fe.fusions3.keys()),
#                            batch_size=1_000_000)

fe.assess_3rd_level_features('5000000', df, all_features)

max_score = 0
best_comb = None

for file in listdir(r'../data'):
    if '_score' in file:
        df_score = pd.read_parquet(fr'../data/{file}')
        if df_score['scores'].max() > max_score:
            max_score = df_score['scores'].max()
            print(file, max_score)
            id_max = np.argmax(df_score['scores'])
            best_comb = df_score.iloc[id_max]

feature1 = all_features[int(best_comb['feature i'])]
feature2 = all_features[int(best_comb['feature j'])]
feature3 = all_features[int(best_comb['feature k'])]
operation = list(fe.fusions3.keys())[int(best_comb['operation'])]

print(f'{feature1} {operation} {feature2} {operation} {feature3}')

print(ghjkl)
targets = ['structural complexity', 'Jv measured [discs/m³]', 'Minkowski']
scores_struct, scores_Jv, scores_mink = utils.assess_fits(
    df, features=all_features, targets=targets)

struct_best, struct_max = utils.get_best_feature(scores_struct, all_features)
Jv_best, Jv_max = utils.get_best_feature(scores_Jv, all_features)
mink_best, mink_max = utils.get_best_feature(scores_mink, all_features)

# print('start 3rd level feature check')

# # save all data as chunk parquets
# # reduce datatype to minimum
# counter = 0
# for l3_f, l3_f_data in fe.gen_3rd_level_features(df_features, all_features):

#     l3_f_data = utils.convert_inf(l3_f_data)
#     if np.isnan(l3_f_data).sum() > 0:
#         # pass if data contains nan
#         pass
#     else:
#         s_struct = utils.assess_fit2(df_features['structural complexity'].values,
#                                       y=l3_f_data, scale_indiv=True)
#         if s_struct > struct_max:
#             struct_max = s_struct
#             struct_best = l3_f
#             print(f'highest struct: {struct_best} with {struct_max}')

#         s_mink = utils.assess_fit2(df_features['Minkowski'].values,
#                                     y=l3_f_data, scale_indiv=True)
#         if s_mink > mink_max:
#             mink_max = s_mink
#             mink_best = l3_f
#             print(f'highest minkowski: {mink_best} with {mink_max}')

#         s_Jv = utils.assess_fit2(df_features['Jv measured [discs/m³]'].values,
#                                   y=l3_f_data, scale_indiv=False)
#         if s_Jv > Jv_max:
#             Jv_max = s_Jv
#             Jv_best = l3_f
#             print(f'highest Jv: {Jv_best} with {Jv_max}')

#     if counter % 10_000 == 0:
#         print(f'{counter} 3rd level features checked')
#     counter += 1

