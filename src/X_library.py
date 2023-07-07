# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:48:55 2023

@author: GEr
"""

from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.measure import block_reduce
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings


class plotter:

    def __init__(self):
        pass

    def Jv_plot(self, df: pd.DataFrame, Jv_s: list,
                limit: float = 100) -> None:
        '''figure that scatters different versions of the computed volumetric
        joint count against each other'''
        fig = plt.figure(figsize=(6, 6))

        ax = fig.add_subplot(1, 1, 1)
        markers = ['o', 'v', 'P', 's', 'X', 'D'] * 2
        for i, jv in enumerate(Jv_s):
            x, y = 'Jv measured [discs/m³]', jv
            r2 = r2_score(df[x], df[y])
            if r2 < 0:
                r2 = '< 0'
            else:
                r2 = round(r2, 2)
            ax.scatter(df[x], df[y], alpha=0.5,
                       label=f'{jv}; R2: {r2}', marker=markers[i])

        ax.set_xlim(left=0, right=limit)
        ax.set_ylim(bottom=0, top=limit)
        ax.grid(alpha=0.5)
        ax.set_xlabel('Jv measured [disc./m³]')
        ax.set_ylabel('Jv computed [disc./m³]')
        ax.plot([0, limit], [0, limit], color='black')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(r'../graphics/JVs.png', dpi=300)
        plt.close()

    def DEM_FEM_data(self, df, p_size=4):
        fig = plt.figure(figsize=(7.87, 5))

        ax = fig.add_subplot(2, 1, 1)
        ax.scatter(df['Jv measured [discs/m³]'], df['structural complexity'],
                   edgecolor='black', color='firebrick', s=p_size, alpha=0.5)
        ax.set_xlim(left=0)
        ax.set_ylabel('structural complexity', color='firebrick')
        ax.grid(alpha=0.5)

        ax2 = ax.twinx()
        ax2.scatter(df['Jv measured [discs/m³]'], df['Minkowski'],
                    edgecolor='black', color='teal', s=p_size, alpha=0.5)
        ax2.set_ylabel('Minkowski dimension', color='teal')
        # for i in range(len(df)):
        #     x = df['Jv measured [discs/m³]'].iloc[i]
        #     y = df['structural complexity'].iloc[i]
        #     ax.text(x, y, s=df.index[i])

        ax = fig.add_subplot(2, 1, 2)
        ax.scatter(df['Jv measured [discs/m³]'], df['avg. RQD'],
                   edgecolor='black', color='firebrick', s=p_size, alpha=0.5)
        ax.set_xlim(left=0)
        ax.set_ylabel('avg. RQD', color='firebrick')
        ax.grid(alpha=0.5)
        ax2 = ax.twinx()
        ax2.scatter(df['Jv measured [discs/m³]'], df['avg. P10'],
                    edgecolor='black', color='teal', s=p_size, alpha=0.5)
        ax2.set_ylabel('avg. P10', color='teal')

        # ax = fig.add_subplot(3, 1, 3)
        # ax.scatter(df['Jv measured [discs/m³]'], df['avg. P21'],
        #            edgecolor='black', color='firebrick', s=p_size, alpha=0.5)
        # ax.set_ylabel('avg. P21', color='firebrick')
        # # ax2 = ax.twinx()
        # # ax2.scatter(df['Jv measured [discs/m³]'], df['P32'],
        # #             edgecolor='black', color='teal', s=p_size, alpha=0.5)
        # # ax2.set_ylabel('P32', color='teal')
        # ax.grid(alpha=0.5)

        ax.set_xlabel('Jv measured [discs/m³]')

        plt.tight_layout()
        plt.savefig(r'../output/data.png', dpi=400)
        plt.close()

    def top_x_barplot(self, values: np.array, labels: np.array, title: str,
                      n_show: int = 10) -> None:
        idx_sort = np.argsort(values)

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.bar(x=np.arange(n_show), height=values[:n_show])
        ax.set_xticks(np.arange(n_show))
        ax.set_xticklabels(labels[:n_show],
                           horizontalalignment='right', rotation=40)
        ax.grid(alpha=0.5)
        ax.set_xlabel(f'{n_show} highest values')
        ax.set_title(title)
        plt.tight_layout()

    def scatter_combinations(self, df: pd.DataFrame,
                             plot_params: list) -> None:
        '''function creates scatter plots of one parameter against another'''
        params_dict = dict(zip(plot_params, list(range(len(plot_params)))))

        log_scale_params = ['avg. app. spacing [m]', 'max block volume [m³]',
                            'avg. block volume [m³]',
                            'avg. block edge length [m]',
                            'n blocks', 'a3', 'a2', 'a1', 'block aspect ratio',
                            'avg. block surface area [m²]', 'Q_struct',
                            'block volume computed [m³]']

        for x, y in list(combinations(plot_params, 2)):
            if df[x].isna().sum() == len(df) or df[y].isna().sum() == len(df):
                pass
            else:
                # fit function to data
                df_tmp = df.dropna(subset=[x, y])
                df_tmp.sort_values(by=x, inplace=True)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(df_tmp[x], df_tmp[y], alpha=0.5)
                for dim in [1, 2, 3]:
                    z_1d = np.polyfit(df_tmp[x].values, df_tmp[y].values, dim)
                    p_1d = np.poly1d(z_1d)
                    function_vals = p_1d(df_tmp[x].values)
                    r2 = round(r2_score(df_tmp[y].values, function_vals), 2)
                    ax.plot(df_tmp[x].values, function_vals,
                            label=f'poly fit {dim} d, r2: {r2}')
                for dim in [1, 2]:
                    z_1d = np.polyfit(df_tmp[x].values, np.log(df_tmp[y].values), dim)
                    p_1d = np.poly1d(z_1d)
                    function_vals = p_1d(df_tmp[x].values)
                    r2 = round(r2_score(df_tmp[y].values, np.exp(function_vals)), 2)
                    ax.plot(df_tmp[x].values, np.exp(function_vals),
                            label=f'log fit {dim} d, r2: {r2}')
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.grid(alpha=0.5)
                if x in log_scale_params:
                    ax.set_xscale('log')
                if y in log_scale_params:
                    ax.set_yscale('log')
                ax.legend()
                plt.tight_layout()
                plt.savefig(fr'../graphics/scatters/{params_dict[x]}_{params_dict[y]}.png', dpi=150)
                plt.close()

    def RQD_spacing_hist_plot(self, df: pd.DataFrame) -> None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 7))

        ax1.hist(df['avg. RQD'], color='grey', edgecolor='black', bins=30)
        ax1.set_title('avg. RQD')

        ax2.hist(df['avg. app. spacing [m]'], color='grey', edgecolor='black',
                 bins=30)
        ax2.set_title('avg. app. spacing [m]')

        plt.tight_layout()
        plt.savefig(r'../graphics/RQD_hist.png', dpi=300)
        plt.close()

    def Pij_plot(self, df: pd.DataFrame, fsize: float = 15) -> None:
        fig = plt.figure(figsize=(9, 7))

        ax = fig.add_subplot(3, 3, 1)
        ax.hist(df['avg. P10'], color='grey', edgecolor='black', bins=30)
        ax.set_title('avg. P10', fontsize=fsize)

        ax = fig.add_subplot(3, 3, 4)
        ax.hist(df['avg. P20'], color='grey', edgecolor='black', bins=30)
        ax.set_title('avg. P20', fontsize=fsize)

        ax = fig.add_subplot(3, 3, 5)
        ax.hist(df['avg. P21'], color='grey', edgecolor='black', bins=30)
        ax.set_title('avg. P21', fontsize=fsize)

        ax = fig.add_subplot(3, 3, 7)
        ax.hist(df['Jv measured [discs/m³]'], color='grey', edgecolor='black',
                bins=30)
        ax.set_title('P30', fontsize=fsize)

        ax = fig.add_subplot(3, 3, 9)
        ax.hist(df['P32'], color='grey', edgecolor='black', bins=30)
        ax.set_title('P32', fontsize=fsize)

        fig.text(x=0.5, y=0.96, s='dimension of measurement', ha='center',
                 fontsize=fsize)
        fig.text(x=0.25, y=0.93, s='0', ha='center', fontsize=fsize)
        fig.text(x=0.55, y=0.93, s='1', ha='center', fontsize=fsize)
        fig.text(x=0.85, y=0.93, s='2', ha='center', fontsize=fsize)

        fig.text(x=0.02, y=0.5, s='dimension of sample', va='center',
                 rotation=90, fontsize=fsize)
        fig.text(x=0.065, y=0.16, s='3D', ha='center', fontsize=fsize)
        fig.text(x=0.065, y=0.45, s='2D', ha='center', fontsize=fsize)
        fig.text(x=0.065, y=0.75, s='1D', ha='center', fontsize=fsize)

        plt.tight_layout(rect=(0.07, 0, 1, 0.93))
        plt.savefig(r'../graphics/Pij_plot.png', dpi=300)
        plt.close()

    def directional_lineplot(self, df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(16, 6))
        params = ['RQD_X', 'P10_X', 'P20_X', 'P21_X', 'apparent spacing_X [m]']
        for i, p in enumerate(params):
            ax = fig.add_subplot(1, 5, i+1)
            cols = [p.replace('_X', f' {direction}') for direction in ['X', 'Y', 'Z']]
            for i in range(len(df)):
                if df['set 1 - type'].iloc[i] == 0:
                    ax.plot([1, 2, 3], df[cols].iloc[i],
                            color='C0', alpha=0.5)
                else:
                    ax.plot([1, 2, 3], df[cols].iloc[i],
                            color='C1', alpha=0.5)
            ax.grid(alpha=0.5)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(['X', 'Y', 'Z'])
            ax.set_ylabel(p.replace('_X', ''))
        plt.tight_layout()
        plt.savefig(r'../graphics/directional_plot.png', dpi=300)
        plt.close()

    def Q_Jv_plot(self, df: pd.DataFrame) -> None:
        x_min, x_max = 0, 125
        y_min, y_max = 0.1, 100

        fig, ax = plt.subplots(figsize=(7, 7))
        cax = ax.scatter(df['Jv measured [discs/m³]'], df['Q_struct'],
                         c=df['avg. RQD'], alpha=0.6, vmin=0, vmax=100,
                         zorder=10)
        ax.hlines([1, 4, 10, 40], xmin=x_min, xmax=x_max, color='black',
                  zorder=1)
        ax.vlines([1, 3, 10, 30, 60], ymin=y_min, ymax=y_max, color='black',
                  zorder=1)

        ax.text(1.1, 0.9, 'very poor', va='top')
        ax.text(1.1, 3.5, 'poor', va='top')
        ax.text(1.1, 9, 'fair', va='top')
        ax.text(1.1, 39, 'good', va='top')
        ax.text(1.1, 99, 'very good', va='top')

        ax.text(0.95, 0.15, 'very large\nblocks', rotation=-90, ha='right')
        ax.text(2.9, 0.15, 'large blocks', rotation=-90, ha='right')
        ax.text(9, 0.15, 'medium-sized\nblocks', rotation=-90, ha='right')
        ax.text(29, 0.15, 'small blocks', rotation=-90, ha='right')
        ax.text(59, 0.15, 'very small\nblocks', rotation=-90, ha='right')
        ax.text(99, 0.15, 'crushed rock', rotation=-90, ha='right')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([1, 3, 10, 30, 60, 100])
        ax.set_yticks([1, 4, 10, 40, 100])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlim(left=x_min, right=x_max)
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.set_xlabel('Jv measured [discs/m³]')
        ax.set_ylabel('Q structural (RQD/Jn)')

        cbar = plt.colorbar(cax)
        cbar.set_label('RQD')

        plt.tight_layout()
        plt.savefig(r'../graphics/Q_Jv_plot.png', dpi=300)
        plt.close()


class math:

    def __init__(self):
        pass

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector, axis=1)[:, None]

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2': """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        dot_products = [np.dot(v1_u[i], v2_u[i]) for i in range(len(v1))]
        angles = np.arccos(dot_products)
        return np.degrees(angles)

    def normal_vectors(self, dips, dipdirs):
        """ convert dip and dip directions into normal vector """
        n_X = np.sin(np.radians(dips)) * np.sin(np.radians(dipdirs))
        n_Y = np.sin(np.radians(dips)) * np.cos(np.radians(dipdirs))
        n_Z = np.cos(np.radians(dips))

        return np.vstack((n_X, n_Y, n_Z)).T


class utilities:

    def __init__(self):
        self.sclr = MinMaxScaler()

    def cust_dropna(self, df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    def convert_inf(self, x: np.array) -> np.array:
        '''function converts all positive and negative infinites to np.nan'''
        return np.where((x == np.inf) | (x == -np.inf), np.nan, x)

    def assess_fit2(self, x: np.array, y: np.array,
                    scale_indiv: bool = False) -> float:
        '''Function assesses how well two sets of parameters fit to each other.
        Assessment is done based on the R2 score. Parameters can be scaled to
        the target or not, depending on the parameter.'''

        y = self.convert_inf(y)
        if np.isnan(y).sum() == 0:
            x = self.sclr.fit_transform(x.reshape(-1, 1))
            if scale_indiv is False:
                y = self.sclr.transform(y.reshape(-1, 1))
            else:
                y = self.sclr.fit_transform(y.reshape(-1, 1))
            x = self.convert_inf(x)
            y = self.convert_inf(y)
            if np.isnan(x).sum() > 0 or np.isnan(y).sum() > 0:
                pass
            else:
                warnings.filterwarnings('ignore')
                score = r2_score(x, y)
            return score

    def assess_fit(self, df, x, y, dropna=True, scale_indiv=False):
        warnings.filterwarnings('ignore')
        df_1 = df[[x, y]]
        if dropna is True:
            df_1 = self.cust_dropna(df_1)
        if len(df_1) < 100:
            score = 2
        else:
            df_1['x_new'] = self.sclr.fit_transform(df_1[x].values.reshape(-1, 1))
            if scale_indiv is False:
                df_1['y_new'] = self.sclr.transform(df_1[y].values.reshape(-1, 1))
            else:
                df_1['y_new'] = self.sclr.fit_transform(df_1[y].values.reshape(-1, 1))
            df_1 = self.cust_dropna(df_1)
            if len(df_1) < 100:
                score = 2
            else:
                score = r2_score(df_1['x_new'], df_1['y_new'])
        return score

    def assess_fits(self, df: pd.DataFrame, features: list,
                    targets: list) -> list:
        scores = []

        n_features = len(features)
        for t in targets:
            print(f'compute {n_features} scores for {t}')
            if t == 'Jv measured [discs/m³]':
                scale_indiv = False
            else:
                scale_indiv = True
            scores_temp = []
            for i, f in enumerate(features):
                scores_temp.append(self.assess_fit(df, x=t, y=f, dropna=True,
                                                   scale_indiv=scale_indiv))
            scores.append(np.array(scores_temp))

        return scores

    def get_best_feature(self, scores, features):
        id_fails = np.where(scores == 2)[0]
        scores = np.delete(scores, id_fails)
        all_features_new = np.delete(np.array(features), id_fails)

        feature_max_score = all_features_new[np.argmax(scores)]
        return feature_max_score, max(scores)

    def voxel2grid(self, voxels, RESOLUTION, color):
        '''function converts an open3d voxel grid into a structured 3D raster
        numpy array'''
        voxels = voxels.get_voxels()
        indices = np.stack(list(vx.grid_index for vx in voxels)).astype('int16')
        colors = np.stack(list(vx.color for vx in voxels)).astype('int8')
        # crop voxels to the bounding box
        id_valid = np.where((np.min(indices, axis=1) >= 0) &
                            (np.max(indices, axis=1) < RESOLUTION))[0]
        indices = indices[id_valid]
        colors = colors[id_valid]
        intersecting = np.where(np.sum(colors, axis=1) == color*3, -1, 1)
        idx = np.lexsort((indices[:,0], indices[:,1], indices[:,2])).reshape(RESOLUTION, RESOLUTION, RESOLUTION)
        return intersecting[idx]


class parameters:

    def __init__(self):
        pass

    def Jv_ISO14689(self, spacings1, spacings2, spacings3):
        '''computes volumetric joint count acc. to EN ISO 14689 and Palmstrøm
        (1982)'''
        set1 = (1/spacings1).fillna(0)
        set2 = (1/spacings2).fillna(0)
        set3 = (1/spacings3).fillna(0)
        return set1 + set2 + set3

    def Jv_Palmstroem2000(self, spacings1, spacings2, spacings3,
                          n_random, tot_volume):
        '''computes volumetric joint count acc. to Palmstrøm (2000)'''
        # TODO check if computation is correct!
        set1 = (1/spacings1).fillna(0)
        set2 = (1/spacings2).fillna(0)
        set3 = (1/spacings3).fillna(0)
        Jv_base = set1 + set2 + set3
        estimated_random = 1  # n_random / tot_volume
        return Jv_base + estimated_random/5

    def Jv_Sonmez1999_1(self, P10x, P10y, P10z):
        '''computes volumetric joint count acc. to Sonmez & Ulusay (1999) for
        three perpendicular scanlines. Original formulation:
            J_v = N_z/L_z + N_y/L_y + N_x/L_x
        where N = number of disc. intersections per scanline and L = length of
        scanline. Here this equals to the definition of P10'''
        return P10x * P10y * P10z

    def Jv_Sonmez1999_2(self, P10_average):
        '''computes volumetric joint count acc. to Sonmez & Ulusay (1999) for
        heavilly frctured rockmasses. Original formulation:
            J_v = (N/L)**3
        where N = number of disc. intersections for one scanline and L = length
        of that scanline. Here this equals to the definition of P10 average'''
        return P10_average**3

    def Jv_Sonmez2002(self, n_disc_sets, average_spacing):
        '''computation of Jv acc. to Sonmez and Ulusay (2002)'''
        return n_disc_sets * (1 / average_spacing)

    def block_volume_palmstroem(self, S1, S2, S3, alpha, beta, gamma):
        '''computation of the average block volume according to Palmstrøm
        (2000)'''
        return S1*S2*S3*(np.sin(np.radians(alpha))*np.sin(np.radians(beta))*np.sin(np.radians(gamma)))

    def compute_n_disc_sets(self, df: pd.DataFrame) -> list:
        '''computes the number of discontinuity sets'''
        sets = ['set_1_ratio', 'set_2_ratio', 'set_3_ratio', 'rand_set_ratio']

        n_s = []  # collection of numbers of discontinuities
        for i in range(len(df)):
            if df['Jv measured [discs/m³]'].iloc[i] < 1:
                # Massive, no or few joints
                n = 0
            elif len(np.where(df.iloc[i][sets].values > 0.15)[0]) == 4:
                # 4 sets
                n = 4
            elif len(np.where(df.iloc[i][sets].values > 0.20)[0]) == 3:
                # 3 sets
                n = 3
            elif len(np.where(df.iloc[i][sets].values > 0.225)[0]) == 2:
                # 2 sets
                n = 2
            elif len(np.where(df.iloc[i][sets].values > 0.40)[0]) == 1:
                # 1 set
                n = 1
            else:
                n = 'n na'
            n_s.append(n)

        return n_s

    def compute_Jn(self, df: pd.DataFrame) -> list:
        '''computes the respective "joint number" rating according to the
        Q-system of '''
        sets = ['set_1_ratio', 'set_2_ratio', 'set_3_ratio', 'rand_set_ratio']

        Jn_s = []  # collection of numbers of discontinuities
        for i in range(len(df)):
            if df['Jv measured [discs/m³]'].iloc[i] < 1:
                # Massive, no or few joints acc. to ISRM blue book
                Jn = 1
            elif df['Jv measured [discs/m³]'].iloc[i] > 60:
                # crushed rock acc. to ISRM blue book
                Jn = 20
            elif len(np.where(df.iloc[i][sets].values > 0.15)[0]) == 4 and df['Jv measured [discs/m³]'].iloc[i] > 30:
                # very small blocks acc. to ISRM blue book
                Jn = 15
            elif len(np.where(df.iloc[i][sets].values > 0.15)[0]) == 4:
                # three joint sets + random
                Jn = 12
            elif len(np.where(df.iloc[i][sets[:3]].values > 0.20)[0]) == 3:
                # three sets
                Jn = 9
            elif df.iloc[i]['rand_set_ratio'] > 0.20 and len(np.where(df.iloc[i][['set_1_ratio', 'set_2_ratio']].values > 0.20)[0]) == 2:
                # two joint sets plus random joints
                Jn = 6
            elif df.iloc[i]['rand_set_ratio'] > 0.20 and len(np.where(df.iloc[i][['set_1_ratio', 'set_3_ratio']].values > 0.20)[0]) == 2:
                # two joint sets plus random joints
                Jn = 6
            elif df.iloc[i]['rand_set_ratio'] > 0.20 and len(np.where(df.iloc[i][['set_2_ratio', 'set_3_ratio']].values > 0.20)[0]) == 2:
                # two joint sets plus random joints
                Jn = 6
            elif len(np.where(df.iloc[i][sets[:3]].values > 0.25)[0]) == 2:
                # two sets
                Jn = 4
            elif df.iloc[i]['rand_set_ratio'] > 0.3 and len(np.where(df.iloc[i][sets[:3]].values > 0.3)[0]) == 1:
                # 1 set plus random joints
                Jn = 3
            elif len(np.where(df.iloc[i][sets].values > 0.4)[0]) == 1:
                # 1 set
                Jn = 2
            else:
                Jn = 'Jn na'
            Jn_s.append(Jn)

        return Jn_s

    def Minkowski(self, n_boxes, box_sizes):
        N_ = np.log(n_boxes)
        eps_ = np.log(1/box_sizes)
        return np.polyfit(eps_, N_, 1)[0]

    def Hausdorff(self, n_boxes, box_sizes):
        log_n_boxs = np.log2(n_boxes)
        log_epsilon = np.log2(box_sizes)
        return -1 * np.polyfit(log_epsilon, log_n_boxs, 1)[0]

    def scalar_p_image(self, arr1, arr2):
        '''function that computs the scalar product of 2 RGB images'''
        return arr1[:, :, 0] * arr2[:, :, 0] + arr1[:, :, 1] * arr2[:, :, 1] + arr1[:, :, 2] * arr2[:, :, 2]

    def structural_complexity(self, data, lambda_=2, N=None, mode='image'):

        if N == None:
            # compute max. possible split with given data resolution
            res = data.shape[0]
            counter = 0
            while res >= 16:
                res /= 2
                counter += 1
            N = counter
            # print(N)

        window_sizes = []
        for _ in range(N):
            window_sizes.append(lambda_)
            lambda_ *= 2

        # collection of stack of coarsened raster data
        stack = [data]

        for i, step_size in enumerate(window_sizes):
            if mode == 'image':
                data_c = block_reduce(data, (step_size, step_size, 1), np.mean)
            elif mode == '3Dgrid':
                data_c = block_reduce(data, (step_size, step_size, step_size), np.mean)
            else:
                raise ValueError('mode not implemented')

            data_c = resize(data_c, data.shape, mode='edge',
                            anti_aliasing=False, anti_aliasing_sigma=None,
                            preserve_range=True, order=0)

            stack.append(data_c)

        overlaps = []  # overlaps between images
        for i in range(len(stack)-1):
            # compute scalar products
            if mode == 'image':
                o1 = self.scalar_p_image(stack[i], stack[i])
                o2 = self.scalar_p_image(stack[i+1], stack[i])
                o3 = self.scalar_p_image(stack[i+1], stack[i+1])
            elif mode == '3Dgrid':
                o1 = stack[i] * stack[i]
                o2 = stack[i+1] * stack[i]
                o3 = stack[i+1] * stack[i+1]
            else:
                raise ValueError('mode not implemented')
            # compute overlap
            overl = np.abs(o2.mean() - 0.5 * (o1.mean() + o3.mean()))
            overlaps.append(overl)
        # compute structural complexity
        complexity = sum(overlaps)
        return complexity
