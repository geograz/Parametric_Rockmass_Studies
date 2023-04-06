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


###########################################
# fixed and constant variables
###########################################

BATCH_SIZE = 10_000_000  # batch size for the third level feature analysis
TARGETS = ['structural complexity', 'Jv measured [discs/m³]', 'Minkowski']
BASE_FEATURES = [  # 'set 1 - radius [m]', 'set 2 - radius [m]',
                 # 'set 3 - radius [m]', 'random set - radius [m]',
                 # 'meas. spacing set 1 [m]', 'meas. spacing set 2 [m]',
                 # 'meas. spacing set 3 [m]',
                 'm_length', 'n_intersections',
                 # 'n_discs',
                 # 'avg. RQD',
                 # 'avg. P10', 'avg. app. spacing [m]',
                 ]
TARGET_3RD_LEVEL = 'struct'  # 'struct', 'mink', 'Jv'

###########################################
# data loading and other preprocessing
###########################################

# instantiation
fe = feature_engineer()
utils = utilities()

# load data
df = pd.read_excel(r'../output/dataset1.xlsx')
df = df.dropna(subset=['structural complexity', 'Minkowski'])
print(len(df))

###########################################
# first and second level feature generation
###########################################

# make first level features
df = fe.make_1st_level_features(df, features=BASE_FEATURES,
                                operations=['log', 'sqrt', 'sqr', 'power_3',
                                            'mult10', 'div10', '1div'],
                                drop_empty=True)
l1_features = [f for f in df.columns if '-l1' in f]

# make second level features
df = fe.make_2nd_level_features(df, features=BASE_FEATURES + l1_features,
                                drop_empty=True)
l2_features = [f for f in df.columns if '-l2' in f]

all_features = BASE_FEATURES + l1_features + l2_features
print(len(all_features))

# check how well all features so far fit to the given targets
scores_struct, scores_Jv, scores_mink = utils.assess_fits(
    df, features=all_features, targets=TARGETS)
struct_best, struct_max = utils.get_best_feature(scores_struct, all_features)
Jv_best, Jv_max = utils.get_best_feature(scores_Jv, all_features)
mink_best, mink_max = utils.get_best_feature(scores_mink, all_features)

###########################################
# third level feature generation
###########################################

# # generate structure of third level features
# fe.gen_3rd_level_structure(all_features, list(fe.fusions3.keys()),
#                             batch_size=BATCH_SIZE)

# compute fitting score for several batches -> can be done in parallel
for batch in ['10000000', '20000000', '30000000']:
    print(f'process batch {batch}')
    fe.assess_3rd_level_features(batch, df, all_features,
                                  target=TARGET_3RD_LEVEL)

# # find highest score
# max_score = 0
# best_comb = None

# for file in listdir(r'../data'):
#     if f'{TARGET_3RD_LEVEL}_score' in file:
#         df_score = pd.read_parquet(fr'../data/{file}')
#         if df_score['scores'].max() > max_score:
#             max_score = df_score['scores'].max()
#             print(file, max_score)
#             id_max = np.argmax(df_score['scores'])
#             best_comb = df_score.iloc[id_max]

# feature1 = all_features[int(best_comb['feature i'])]
# feature2 = all_features[int(best_comb['feature j'])]
# feature3 = all_features[int(best_comb['feature k'])]
# operation = list(fe.fusions3.keys())[int(best_comb['operation'])]

# print(TARGET_3RD_LEVEL, max_score)
# print(f'{feature1} {operation} {feature2} {operation} {feature3}')
