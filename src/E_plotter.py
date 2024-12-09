# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:52:41 2024

@author: GEr
"""

import pandas as pd

from X_library import plotter


##########################################
# static variables and constants
##########################################

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

df = pd.read_excel(r'../output/PDD1_1.xlsx', index_col='identifier')

##########################################
# visualizations of the dataset
##########################################

pltr.Euler_plot()

pltr.advanced_parameter_plot(df)

pltr.complexity_scatter(df)

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
