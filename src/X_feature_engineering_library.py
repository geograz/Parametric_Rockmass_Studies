# -*- coding: utf-8 -*-
"""
script that generates new features based on simple mathematical operations on a
dataframe

author: Georg H. Erharter (georg.erharter@ngi.no)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from X_library import utilities


class operations:

    def __init__(self):
        pass

    def power_3(self, x):
        return x**3

    def multiply2(self, x):
        return x*2

    def multiply3(self, x):
        return x*3

    def multiply10(self, x):
        return x*10

    def div2(self, x):
        return x/2

    def div10(self, x):
        return x/10

    def _1div(self, x):
        return 1/x

    def add_2(self, x, y):
        return x + y

    def subtract_2(self, x, y):
        return x - y

    def multiply_2(self, x, y):
        return x * y

    def divide_2(self, x, y):
        return x / y

    def power(self, x, y):
        return x**y

    def add_3(self, x, y, z):
        return x + y + z

    def minus_3(self, x, y, z):
        return x - y - z

    def multiply_3(self, x, y, z):
        return x * y * z

    def div_3(self, x, y, z):
        return x / y / z


class feature_engineer(operations):

    def __init__(self):

        self.utils = utilities()

        # collection of operations to apply to make first level features
        self.transformations = {'log': np.log, 'sqrt': np.sqrt,
                                'sqr': np.square, 'exp': np.exp,
                                'power_3': self.power_3,
                                'mult2': self.multiply2,
                                'mult3': self.multiply3,
                                'mult10': self.multiply10,
                                'div2': self.div2, 'div10': self.div10,
                                '1div': self._1div}
        # collection of operations to apply to make second level features
        self.fusions2 = {'plus': self.add_2, 'minus': self.subtract_2,
                         'times': self.multiply_2, 'dividedby': self.divide_2,
                         'power': self.power}
        # collection of operations to apply to make third level features
        self.fusions3 = {'plus': self.add_3, 'times': self.multiply_3,
                         'minus': self.minus_3, 'dividedby': self.div_3}
        # operations that are commutative e.g. a + b = b + a -> redundant
        self.commutative = ('plus', 'times')

    def drop_no_information_cols(self, df: pd.DataFrame,
                                 nan_threshold: int = 300) -> pd.DataFrame:
        '''function removes columns from dataframe without information or too
        many NAN'''
        id_0 = np.where(df.sum(axis=0).values == 0)[0]
        df.drop(columns=df.columns[id_0], inplace=True)
        id_nan = np.where(df.isna().sum().values > nan_threshold)[0]
        df.drop(columns=df.columns[id_nan], inplace=True)
        return df

    def gen_3rd_level_structure(self, features, operations, batch_size):
        n_features = tuple(range(len(features)))
        n_operations = tuple(range(len(operations)))

        tot_combs = len(features)*len(features)*len(features)*len(operations)
        print(f'{round(tot_combs / 1_000_000_000, 3)} billion combinations')
        print(f'{int(tot_combs/batch_size)} files will be saved')

        counter = 0
        i_s, j_s, k_s, l_s = [], [], [], []
        while counter < batch_size:
            for i in n_features:
                for j in n_features:
                    for k in n_features:
                        for l in n_operations:
                            i_s.append(i)
                            j_s.append(j)
                            k_s.append(k)
                            l_s.append(l)
                            counter += 1
                            if counter % batch_size == 0 or counter == tot_combs-1:
                                df = pd.DataFrame({'feature i': np.array(i_s).astype(np.int16),
                                                   'feature j': np.array(j_s).astype(np.int16),
                                                   'feature k': np.array(k_s).astype(np.int16),
                                                   'operation': np.array(l_s).astype(np.int8)})
                                df.to_parquet(fr'../features/{counter}.gzip',
                                              index=False)
                                i_s.clear()
                                j_s.clear()
                                k_s.clear()
                                l_s.clear()

    def assess_3rd_level_features(self, filename, df, features, target,
                                  savepath):
        df_indices = pd.read_parquet(fr'../features/{filename}.gzip')
        operations = list(self.fusions3.keys())

        scores = []
        for i in tqdm(range(len(df_indices))):
            x, y, z, f = df_indices.iloc[i]
            x, y, z, f = features[x], features[y], features[z], operations[f]
            new_feature = self.fusions3[f](df[x], df[y], df[z]).values

            new_feature = self.utils.convert_inf(new_feature)
            if np.isnan(new_feature).sum() > 0:
                # pass if data contains nan
                score = np.nan
            else:
                if target == 'struct':
                    score = self.utils.assess_fit2(
                        df['structural complexity'].values, y=new_feature,
                        scale_indiv=True)
                elif target == 'mink':
                    score = self.utils.assess_fit2(
                        df['Minkowski'].values, y=new_feature,
                        scale_indiv=True)
                elif target == 'Jv':
                    score = self.utils.assess_fit2(
                        df['Jv measured [discs/m³]'].values, y=new_feature,
                        scale_indiv=False)
                elif target == 'P32':
                    score = self.utils.assess_fit2(
                        df['P32'].values, y=new_feature,
                        scale_indiv=False)
                else:
                    raise ValueError(f'{target} as target not available!')
            scores.append(score)
        df_indices['scores'] = np.array(scores).astype(np.float32)
        # only save results with a score > R2 = 0.5 to save space
        df_indices = df_indices[df_indices['scores'] > 0.5]
        df_indices.to_parquet(savepath, index=False)

    def make_1st_level_features(self, df, features: list = None,
                                operations: list = None,
                                drop_empty: bool = False) -> pd.DataFrame:
        '''function that computes new features based on single features only.
        Unless specified the function computes new features for all columns of
        the dataframe and also uses all possible operations.'''
        if features is None:
            features = df.columns
        if operations is None:
            operations = self.transformations.keys()

        new_headers = []
        new_cols = []
        for f in features:
            for t in operations:
                new_headers.append(f'{t}_{f}-l1')
                new_cols.append(self.transformations[t](df[f]))
        df_temp = pd.DataFrame(columns=new_headers,
                               data=np.array(new_cols).T,
                               index=df.index)
        df = pd.concat([df, df_temp], axis=1)

        if drop_empty is True:
            df = self.drop_no_information_cols(df)
        print('level 1 features computed', len(df.columns))
        return df

    def make_2nd_level_features(self, df, features: list = None,
                                operations: list = None,
                                drop_empty: bool = False) -> pd.DataFrame:
        '''function that computes new features based on all unique combinations
        of 2 features in the dataframe.
        Unless specified the function computes new features for all columns of
        the dataframe and also uses all possible operations.'''
        if features is None:
            features = df.columns
        if operations is None:
            operations = self.fusions2.keys()

        new_headers = []
        new_cols = []
        for i, x in enumerate(features):
            for j, y in enumerate(features):
                if i == j:  # avoid duplicates
                    pass
                else:
                    for f in operations:
                        if f in self.commutative and j > i:
                            # avoid making duplicate features due to
                            # commutative operations
                            pass
                        else:
                            new_headers.append(f'{x}_{f}_{y}-l2')
                            new_cols.append(self.fusions2[f](df[x], df[y]))
        # check if duplicates were generated
        if len(new_headers) != len(set(new_headers)):
            raise ValueError('duplicate features generated in second level')
        else:
            df_temp = pd.DataFrame(columns=new_headers,
                                   data=np.array(new_cols).T,
                                   index=df.index)
            df = pd.concat([df, df_temp], axis=1)

            if drop_empty is True:
                df = self.drop_no_information_cols(df)
            print('level 2 features computed', len(df.columns))
            return df

    def get_top_n_scores_in_files(self, n, file_paths):
        '''function iterates files with scores and only saves the top n scores
        and combinations for further analyses.
        original function made by ChatGPT'''
        top_scores = []
        for file_path in file_paths:
            df = pd.read_parquet(file_path)
            for idx, row in df.iterrows():
                score = row['scores']
                if len(top_scores) < n:
                    top_scores.append((score, row))
                    top_scores.sort(reverse=True, key=lambda x: x[0])
                elif score > top_scores[-1][0]:
                    top_scores[-1] = (score, row)
                    top_scores.sort(reverse=True, key=lambda x: x[0])
        return [x[1] for x in top_scores]

    def decode_combination(self, combination: pd.Series, all_features: list):
        f1 = all_features[int(combination['feature i'])]
        f2 = all_features[int(combination['feature j'])]
        f3 = all_features[int(combination['feature k'])]
        operation = list(self.fusions3.keys())[int(combination['operation'])]
        comb_string = f'{f1} {operation} {f2} {operation} {f3}'
        return comb_string


# example usage of code
if __name__ == '__main__':

    # create dummy data
    df = pd.DataFrame({'x': np.random.uniform(0, 1, 100),
                       'y': np.random.uniform(0, 1, 100),
                       'z': np.random.uniform(0, 1, 100)})

    # instantiate feature engineer
    fe = feature_engineer()
    # create first level features
    df = fe.make_1st_level_features(df, features=['x', 'y'],
                                    operations=['log', 'sqrt'])
    # create second level features of all previous ones
    df = fe.make_2nd_level_features(df)
