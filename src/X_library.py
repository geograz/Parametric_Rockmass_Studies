# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:48:55 2023

@author: GEr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.measure import block_reduce
from sklearn.metrics import r2_score


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
        plt.savefig(r'../graphics/_JVs.png')
        plt.close()

    def DEM_FEM_data(self, df):
        fig = plt.figure(figsize=(7.87, 7.87))

        ax = fig.add_subplot(3, 1, 1)
        ax.scatter(df['Jv measured [discs/m³]'], df['structural complexity'],
                   edgecolor='black', color='grey', alpha=0.5)
        # for i in range(len(df)):
        #     x = df['Jv measured [discs/m³]'].iloc[i]
              # y = df['structural complexity'].iloc[i]
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
        pass

    def min_max_scaler(self, x):
        x = x-x.min()
        x = x/x.max()
        return x

    def assess_fit(self, df, x, y, dropna=False):
        df_1 = df[[x, y]]
        if dropna is True:
            df_1.dropna(inplace=True)
        df_1['x_new'] = self.min_max_scaler(df_1[x])
        df_1['y_new'] = self.min_max_scaler(df_1[y])
        if dropna is True:
            df_1.dropna(inplace=True)
        if len(df_1) < 100:
            score = 2
        else:
            score = r2_score(df_1['x_new'], df_1['y_new'])
        return score


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

    def Qsys_Jn(self, set_1_ratio, set_2_ratio, set_3_ratio, rand_set_ratio,
                n_tot):

        Jn_s = []
        for i in range(len(n_tot)):
            if n_tot[i] < 100:
                Jn = 1
            elif set_1_ratio[i] > 0.14 and set_2_ratio[i] > 0.14 and set_3_ratio[i] > 0.14 and rand_set_ratio[i] > 0.14:
                # 3 sets + random
                Jn = 12
            elif set_1_ratio[i] > 0.2 and set_2_ratio[i] > 0.2 and set_3_ratio[i] > 0.2:
                # 3 sets
                Jn = 9
            elif (set_1_ratio[i] > 0.18 and set_2_ratio[i] > 0.18) or (set_1_ratio[i] > 0.18 and set_3_ratio[i] > 0.18) or (set_2_ratio[i] > 0.18 and set_3_ratio[i] > 0.18) and rand_set_ratio[i] > 0.22:
                # 2 sets + random
                Jn = 6
            elif (set_1_ratio[i] > 0.25 and set_2_ratio[i] > 0.25) or (set_1_ratio[i] > 0.25 and set_3_ratio[i] > 0.25) or (set_2_ratio[i] > 0.25 and set_3_ratio[i] > 0.25):
                # 2 sets
                Jn = 4
            elif set_1_ratio[i] > 0.35 or set_2_ratio[i] > 0.35 or set_3_ratio[i] > 0.35 and rand_set_ratio[i] > 0.25:
                # 1 set + random
                Jn = 3
            elif set_1_ratio[i] > 0.35 or set_2_ratio[i] > 0.35 or set_3_ratio[i] > 0.35 or rand_set_ratio[i] > 0.35:
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
            while res >= 8:
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
            # print(f'coarsening {i} done')

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
            overl = o2.mean() - 0.5 * (o1.mean() + o3.mean())
            overlaps.append(overl)
        # compute structural complexity
        complexity = np.abs(sum(overlaps))
        # print(f'C: {complexity}')
        return complexity
