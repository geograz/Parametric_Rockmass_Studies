# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:48:55 2023

@author: GEr
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.measure import block_reduce


class plotter:

    def __init__(self):
        pass

    def custom_scatter(self, df, x, y, color_feature=None,
                       annotation=False, f_size=5, p_size=.5,
                       scale_max=False):
        ax = plt.gca()
        if color_feature == None:
            ax.scatter(df[x], df[y], s=p_size, edgecolors='none', alpha=0.5)
        else:
            ax.scatter(df[x], df[y], c=df[color_feature], s=p_size,
                       edgecolors='none', alpha=0.5)
        if annotation is True:
            df_temp = df.dropna(subset='Hausdorff')
            for i, label in enumerate(df_temp.index):
                ax.annotate(label, (df_temp[x].iloc[i], df_temp[y].iloc[i]))
        ax.set_xlabel(x, fontsize=f_size)
        ax.set_ylabel(y, fontsize=f_size)
        ax.tick_params(axis='both', labelsize=f_size)
        ax.grid(alpha=0.5, linewidth=.3)
        if scale_max is True:
            max_ = max(df[[x, y]].max())
            min_ = min(df[[x, y]].min())
            ax.set_xlim(left=min_, right=max_)
            ax.set_ylim(bottom=min_, top=max_)

    def mass_scatter(self, df, plot_params, savepath):
        fig = plt.figure(figsize=(14, 14))

        font_size = 2.5
        line_width = 0.3
        p_size = .2

        n_plot_params = len(plot_params)
        idx = 1

        for i in range(n_plot_params):
            for j in range(n_plot_params):

                ax = fig.add_subplot(n_plot_params, n_plot_params, idx)
                if i == j:
                    ax.hist(df[plot_params[i]].dropna(), bins=20,
                            edgecolor='black', linewidth=line_width)
                    ax.set_xlabel(plot_params[i], fontsize=font_size)
                else:
                    ax.scatter(df[plot_params[i]], df[plot_params[j]], s=p_size,
                               edgecolors='none', alpha=0.5)

                    log_scale_params = ['avg. app. spacing [m]', 'max block volume [m³]',
                                        'avg. block volume [m³]', 'avg. block edge length [m]',
                                        'n blocks', 'a3', 'a2', 'a1', 'block aspect ratio',
                                        'avg. block surface area [m²]', 'Q_struct']
                    if plot_params[i] in log_scale_params:
                        ax.set_xscale('log')
                    if plot_params[j] in log_scale_params:
                        ax.set_yscale('log')
                    ax.set_xlabel(plot_params[i], fontsize=font_size)
                    ax.set_ylabel(plot_params[j], fontsize=font_size)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticks([], minor=True)
                ax.set_yticks([], minor=True)

                idx += 1

        plt.minorticks_off()

        plt.tight_layout(h_pad=1, w_pad=1)
        plt.savefig(savepath, dpi=1200)
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
