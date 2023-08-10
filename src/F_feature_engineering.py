# -*- coding: utf-8 -*-
"""
Experimental code to the paper XXXX
Dr. Georg H. Erharter - 2023
DOI: XXXXXXXXXXX

Script that investigates different ways of developing new parameters for rock
mass characterization
(results not included in first paper).
"""

import numpy as np
import pandas as pd
from os import listdir

from X_library import utilities, plotter
from X_feature_engineering_library import feature_engineer


###########################################
# fixed and constant variables
###########################################

BATCH_SIZE = 10_000_000  # batch size for the third level feature analysis
TARGETS = ['structural complexity', 'Jv measured [discs/m³]', 'Minkowski',
           'P32']
BASE_FEATURES = [  # 'set 1 - radius [m]', 'set 2 - radius [m]',
                 # 'set 3 - radius [m]', 'random set - radius [m]',
                 # 'meas. spacing set 1 [m]', 'meas. spacing set 2 [m]',
                 # 'meas. spacing set 3 [m]',
                 'm_length', 'n_intersections',
                 # 'n_discs',
                 # 'avg. RQD',
                 # 'avg. P10', 'avg. app. spacing [m]',
                 ]
TARGET_3RD_LEVEL = 'Jv'  # 'struct', 'mink', 'Jv', 'P32'
MODE = 'evaluation'  # 'structure', 'scores', 'evaluation'
N_TOP_SCORES = 20

###########################################
# data loading and other preprocessing
###########################################

# instantiation
fe = feature_engineer()
utils = utilities()
pltr = plotter()

# load data
df = pd.read_excel(r'../output/dataset1.xlsx')
print(len(df))

###########################################
# first and second level feature generation
###########################################

# make first level features
df = fe.make_1st_level_features(df, features=BASE_FEATURES,
                                operations=['log', 'sqrt', 'sqr', 'power_3',
                                            'mult10', 'div10', '1div', 'div2'],
                                drop_empty=True)
l1_features = [f for f in df.columns if '-l1' in f]

# make second level features
df = fe.make_2nd_level_features(df, features=BASE_FEATURES + l1_features,
                                drop_empty=True)
l2_features = [f for f in df.columns if '-l2' in f]

all_features = BASE_FEATURES + l1_features + l2_features
print(len(all_features))

# # check how well all features so far fit to the given targets
# scores_struct, scores_Jv, scores_mink, scores_P32 = utils.assess_fits(
#     df, features=all_features, targets=TARGETS)
# struct_best, struct_max = utils.get_best_feature(scores_struct, all_features)
# Jv_best, Jv_max = utils.get_best_feature(scores_Jv, all_features)
# mink_best, mink_max = utils.get_best_feature(scores_mink, all_features)
# P32_best, P32_max = utils.get_best_feature(scores_P32, all_features)

# print(f'struct: {round(struct_max, 3)}, Jv: {round(Jv_max, 3)}, Mink: {round(mink_max, 3)}, P32: {round(P32_max, 3)}')

###########################################
# third level feature processing
###########################################

# generate all possible combinations of third level features
if MODE == 'structure':
    fe.gen_3rd_level_structure(all_features, list(fe.fusions3.keys()),
                               batch_size=BATCH_SIZE)
# compute fitting score for several batches -> can be done in parallel
elif MODE == 'scores':
    for batch in np.arange(90000000, 1000000000, step=10_000_000)[0:]:
        print(f'process batch {batch}')
        filename = f'{batch}_{TARGET_3RD_LEVEL}_score.gzip'
        if filename in listdir(r'../features'):
            print(f'{filename} already processed -> skip')
            pass
        else:
            print(f'process {filename}')
            savepath = fr'../features/{filename}'
            fe.assess_3rd_level_features(batch, df, all_features,
                                         target=TARGET_3RD_LEVEL,
                                         savepath=savepath)
# find best performing combinations of parameters
elif MODE == 'evaluation':

    filepaths = [fr'../features/{f}' for f in listdir(r'../features') if f'{TARGET_3RD_LEVEL}_score' in f]

    result = fe.get_top_n_scores_in_files(n=N_TOP_SCORES, file_paths=filepaths)

    best_comb = result[0]

    best_comb_equation = fe.decode_combination(best_comb, all_features)

    print(TARGET_3RD_LEVEL, best_comb['scores'])
    print(best_comb_equation)

    pltr.top_x_barplot(values=[r['scores'] for r in result],
                       labels=[fe.decode_combination(r, all_features) for r in result],
                       title='test')
