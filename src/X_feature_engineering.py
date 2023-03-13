# -*- coding: utf-8 -*-
"""
script that generates new features based on simple mathematical operations on a
dataframe

author: Georg H. Erharter (georg.erharter@ngi.no)
"""

import numpy as np
import pandas as pd


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

    def multiply_3(self, x, y, z):
        return x * y * z


class feature_engineer(operations):

    def __init__(self):

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
        self.fusions3 = {'plus': self.add_3, 'times': self.multiply_3}
        # operations that are commutative e.g. a + b = b + a -> redundant
        self.commutative = ['plus', 'times']

    def drop_no_information_cols(self, df: pd.DataFrame,
                                 nan_threshold: int = 300) -> pd.DataFrame:
        '''function removes columns from dataframe without information or too
        many NAN'''
        id_0 = np.where(df.sum(axis=0).values == 0)[0]
        df.drop(columns=df.columns[id_0], inplace=True)
        id_nan = np.where(df.isna().sum().values > nan_threshold)[0]
        df.drop(columns=df.columns[id_nan], inplace=True)
        return df

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

            return df

    def make_3rd_level_features(self, df, features: list = None,
                                operations: list = None,
                                drop_empty: bool = False) -> pd.DataFrame:
        '''function that computes new features based on all unique combinations
        of 3 features in the dataframe.
        Unless specified the function computes new features for all columns of
        the dataframe and also uses all possible operations.'''
        if features is None:
            features = df.columns
        if operations is None:
            operations = self.fusions3.keys()

        new_headers = []
        new_cols = []
        for i, x in enumerate(features):
            for j, y in enumerate(features):
                for k, z in enumerate(features):
                    if i == j or i == k or j == k:  # avoid duplicates
                        pass
                    else:
                        for f in operations:
                            if j > i or k > i or k > j:
                                # avoid making duplicate features due to
                                # commutative operations
                                pass
                            else:
                                new_headers.append(f'{x}_{f}_{y}_{f}_{z}-l3')
                                new_cols.append(self.fusions3[f](df[x], df[y], df[z]))
        # check if duplicates were generated
        if len(new_headers) != len(set(new_headers)):
            raise ValueError('duplicate features generated in third level')
        else:
            df_temp = pd.DataFrame(columns=new_headers,
                                   data=np.array(new_cols).T,
                                   index=df.index)
            df = pd.concat([df, df_temp], axis=1)

            if drop_empty is True:
                df = self.drop_no_information_cols(df)

            return df


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
    # create third level features of basic + level 1 features only
    feature_subset = [f for f in df.columns if '-l2' not in f]
    df = fe.make_3rd_level_features(df, features=feature_subset)
