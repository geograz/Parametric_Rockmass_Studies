# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:48:55 2023

@author: GEr
"""

from itertools import combinations
import matplotlib.pyplot as plt
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
        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(1, 1, 1)

        for i, jv in enumerate(Jv_s):
            x, y = 'Jv measured [discs/m³]', jv
            r2 = r2_score(df[x], df[y])
            ax.scatter(df[x], df[y], alpha=0.5,
                       label=f'{jv}; R2: {round(r2, 2)}')

        ax.set_xlim(left=0, right=limit)
        ax.set_ylim(bottom=0, top=limit)
        ax.grid(alpha=0.5)
        ax.set_xlabel(x)
        ax.set_ylabel('Jv computed')
        ax.plot([0, limit], [0, limit], color='black')
        ax.legend()

        plt.tight_layout()
        plt.savefig(r'../graphics/JVs.png')
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

        fig, ax = plt.subplots()
        ax.bar(x=np.arange(n_show), height=values[idx_sort][-n_show:])
        ax.set_xticks(np.arange(n_show))
        ax.set_xticklabels(labels[idx_sort][-n_show:],
                           horizontalalignment='right', rotation=40)
        ax.grid(alpha=0.5)
        ax.set_xlabel(f'{n_show} highest values')
        ax.set_title(title)
        plt.tight_layout()

    def scatter_combinations(self, df: pd.DataFrame,
                             plot_params: list) -> None:
        params_dict = dict(zip(plot_params, list(range(len(plot_params)))))

        log_scale_params = ['avg. app. spacing [m]', 'max block volume [m³]',
                            'avg. block volume [m³]',
                            'avg. block edge length [m]',
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

    def convert_inf(self, x):
        x = np.where((x == np.inf) | (x == -np.inf), np.nan, x)
        return x

    def assess_fit2(self, x: np.array, y: np.array,
                    scale_indiv: bool = False) -> float:

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
        '''computes volumetric joint count acc. to EN ISO 14689'''
        set1 = (1/spacings1).fillna(0)
        set2 = (1/spacings2).fillna(0)
        set3 = (1/spacings3).fillna(0)
        return set1 + set2 + set3

    def Jv_Palmstroem2005(self, spacings1, spacings2, spacings3,
                          n_random, tot_volume):
        '''computes volumetric joint count acc. to Palmstrøm 2005'''
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
        return P10x + P10y + P10z

    def Jv_Sonmez1999_2(self, P10_average):
        '''computes volumetric joint count acc. to Sonmez & Ulusay (1999) for
        heavilly frctured rockmasses. Original formulation:
            J_v = (N/L)**3
        where N = number of disc. intersections for one scanline and L = length
        of that scanline. Here this equals to the definition of P10 average'''
        return P10_average**3

    def Sonmez2002(self):
        pass

    def block_volume_palmstroem(self, S1, S2, S3, alpha, beta, gamma):
        return S1*S2*S3*(np.sin(np.radians(alpha))*np.sin(np.radians(beta))*np.sin(np.radians(gamma)))

    def compute_n_disc_sets(self, set_1_ratio: np.array, set_2_ratio: np.array,
                            set_3_ratio: np.array, rand_set_ratio: np.array,
                            n_tot: np.array) -> list:
        '''computes the number of discontinuity sets and the respective "joint
        number" rating according to the Q-system'''

        n_s = []  # collection of numbers of discontinuities
        Jn_s = []  # collection of J_n from the Q-system
        for i in range(len(n_tot)):
            if n_tot[i] < 100:
                # Massive, no or few joints
                n, Jn = 0, 1
            elif set_1_ratio[i] > 0.14 and set_2_ratio[i] > 0.14 and set_3_ratio[i] > 0.14 and rand_set_ratio[i] > 0.14:
                # 3 sets + random
                n, Jn = 4, 12
            elif set_1_ratio[i] > 0.2 and set_2_ratio[i] > 0.2 and set_3_ratio[i] > 0.2:
                # 3 sets
                n, Jn = 3, 9
            elif (set_1_ratio[i] > 0.18 and set_2_ratio[i] > 0.18) or (set_1_ratio[i] > 0.18 and set_3_ratio[i] > 0.18) or (set_2_ratio[i] > 0.18 and set_3_ratio[i] > 0.18) and rand_set_ratio[i] > 0.22:
                # 2 sets + random
                n, Jn = 3, 6
            elif (set_1_ratio[i] > 0.25 and set_2_ratio[i] > 0.25) or (set_1_ratio[i] > 0.25 and set_3_ratio[i] > 0.25) or (set_2_ratio[i] > 0.25 and set_3_ratio[i] > 0.25):
                # 2 sets
                n, Jn = 2, 4
            elif set_1_ratio[i] > 0.35 or set_2_ratio[i] > 0.35 or set_3_ratio[i] > 0.35 and rand_set_ratio[i] > 0.25:
                # 1 set + random
                n, Jn = 2, 3
            elif set_1_ratio[i] > 0.35 or set_2_ratio[i] > 0.35 or set_3_ratio[i] > 0.35 or rand_set_ratio[i] > 0.35:
                # 1 set
                n, Jn = 1, 2
            else:
                n, Jn = 'n na', 'Jn na'
            n_s.append(n)
            Jn_s.append(Jn)

        return n_s, Jn_s

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
