# -*- coding: utf-8 -*-
"""
        PARAMETRIC ROCK MASS STUDIES
-- computational rock mass characterization --

Code author: Dr. Georg H. Erharter

Script that contains a custom library with different classes of functions for
math, general use (utilities), plotting and computation of parameters.
"""

import os
import zlib

import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter, label
from scipy.optimize import curve_fit
from scipy.stats import entropy
from skimage.transform import resize
from skimage.measure import block_reduce, euler_number
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import trimesh
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class math:
    '''class with mathematical functions'''

    def __init__(self):
        pass

    def unit_vector(self, vector):
        '''compute the unit vector of another vector'''
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
    '''class with miscaleneous functions that don't fit anywhere else'''

    def __init__(self):
        self.sclr = MinMaxScaler()

    def power_law(self, x, a, b):
        return a * np.power(x, b)

    def power_law_neg(self, x, a, b, c):
        return a * np.power(x, b) + c

    def exponential(self, x, a, b, c):
        return a * np.exp(-b*x) * (b*x + c)

    def fit_power_law_neg(self, x, y):
        # Perform the curve fit
        p0 = [-0.05, 1.01, 3]
        params, _ = curve_fit(self.power_law_neg, x, y, p0=p0)

        # Extract the fitted parameters
        a_fit, b_fit, c_fit = params
        return a_fit, b_fit, c_fit

    def fit_power_law(self, x, y):
        # Perform the curve fit
        params, _ = curve_fit(self.power_law, x, y)

        # Extract the fitted parameters
        a_fit, b_fit = params
        return a_fit, b_fit

    def fit_exponential(self, x, y):
        # Perform the curve fit
        p0 = [100, 0.1, 1]
        params, _ = curve_fit(self.exponential, x, y, p0=p0)

        # Extract the fitted parameters
        a_fit, b_fit, c_fit = params
        return a_fit, b_fit, c_fit

    def array_to_pointcloud(self, array: np.array, resolution: float,
                            savepath: str) -> None:
        '''function takes a structured array and computes the coordinates for
        each point and saves them to a zipped csv file'''
        n = array.shape[0]
        x = np.arange(0, n * resolution, resolution)
        y = np.arange(0, n * resolution, resolution)
        z = np.arange(0, n * resolution, resolution)
        # Create the 3D grid of coordinates
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        x_flat = X.ravel()
        y_flat = Y.ravel()
        z_flat = Z.ravel()
        data_flat = array.ravel()
        output = np.column_stack((x_flat, y_flat, z_flat, data_flat))

        df = pd.DataFrame(data=output, columns=['x', 'y', 'z', 'v'])
        df = df.astype({'x': 'float16', 'y': 'float16', 'z': 'float16',
                        'v': 'int32'})
        compression_options = dict(method='zip', archive_name='blocks.csv')
        df.to_csv(savepath, index=False, compression=compression_options)

    def identify_intact_rock_regions(self, array: np.array) -> list:
        """ Identifies all disconnected regions of intact rock (value 0) in a
        3D binary array.
        Parameters:
        - array: 3D numpy array of binary values (0 and 1)
        Returns:
        - labeled_array: 3D array with unique labels for each connected region
        of 0s
        - num_features: Total number of disconnected regions identified"""
        # Define the structure for connectivity
        connectivity = np.array([[[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]],
                                 [[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]],
                                 [[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]]], dtype=int)

        # Perform connected component labeling
        labeled_array, num_features = label(array == 0, structure=connectivity)

        return labeled_array, num_features

    def make_rel_fp(self, path: str, levels_up: int = 0) -> str:
        '''function creates a relative filepath from the executing python file
        location'''
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if levels_up > 0:
            lvls = ['..'] * levels_up
            path_list = [script_dir] + lvls + [path]
        else:
            path_list = [script_dir, path]
        return os.path.abspath(os.path.join(*path_list))




class plotter(utilities):
    '''class with plotting functions'''

    def __init__(self):
        self.label_map = {
            'Jv measured [discs/m³]': 'volumetric joint count - $J_v$ [discontinuities/volume]',
            'P32': '$P32$',
            'Shannon entropy': 'Shannon entropy ($H$)',
            'compression ratio': 'Compression complexity ($C_c$)',
            'structural complexity': 'Multiscale structural complexity ($C$)',
            'Euler characteristic': 'Euler characteristic ($χ$)\nfor discontinuities',
            'Minkowski dimension': 'Fractal dimension ($D$)',
            'Euler characteristic inverted': 'Euler characteristic($χ$)\nfor intact rock'}

    def complexity_scatter(self, df: pd.DataFrame,
                           x_param: str = 'Jv measured [discs/m³]',
                           y_param: str = 'structural complexity',
                           save_high_density: bool = False) -> None:
        '''plots discontinuity density parameter against complexity parameter
        and shows examples for final interpretation'''

        # find examples of samples to add to plot
        df_new = df.dropna(axis=0, subset=[x_param, y_param])
        id_lower = df_new.index[df_new[x_param].argmin()]
        id_upper = df_new.index[df_new[x_param].argmax()]
        id_max_c = df_new.index[df_new[y_param].argmax()]
        Jv_at_max_c = df_new.loc[id_max_c, x_param]
        c_lower_mid = df_new.loc[[id_lower, id_max_c], y_param].mean()
        Jv_upper_mid = df_new.loc[[id_max_c, id_upper], x_param].mean()
        Jv_upper_mid = 50
        id_lower_mid = (df_new[df_new[x_param] < Jv_at_max_c][y_param] - c_lower_mid).abs().idxmin()
        id_upper_mid = (df_new[df_new[x_param] > Jv_at_max_c][x_param] - Jv_upper_mid).abs().idxmin()
        ids = [id_lower, id_lower_mid, id_max_c, id_upper_mid, id_upper]
        print(ids)
        print(df_new[x_param].max())

        # make plot
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(nrows=2, ncols=5, height_ratios=[3.0, 1])

        # top part with complexity
        ax = fig.add_subplot(gs[0, :])
        ax.scatter(df_new[x_param], df_new[y_param],
                   color='dimgrey', alpha=0.3, s=10)
        ax.scatter(df_new.loc[ids, x_param], df_new.loc[ids, y_param],
                   color='grey', edgecolor='black', s=90, zorder=100)
        ax.set_xticks([0, 3, 10, 30, 100])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.tick_top()
        text_y = df_new[y_param].max() + (df_new[y_param].max() - df_new[y_param].min()) * 0.04
        ax.text(x=1, y=text_y, s='extremely low - low', ha='center',
                va='top', rotation=90, color='black')
        ax.text(x=4, y=text_y, s='moderately high', ha='center', va='top',
                rotation=90, color='black')
        ax.text(x=11, y=text_y, s='high', ha='center', va='top',
                rotation=90, color='black')
        ax.text(x=31, y=text_y, s='very high', ha='center', va='top',
                rotation=90, color='black')
        ax.text(x=101, y=text_y, s='extremely high (crushed)', ha='center',
                va='top', rotation=90, color='black')
        ax.set_xlabel(self.label_map[x_param])
        ax.xaxis.set_label_position('top')
        ax.set_ylabel(f'Rock Mass Complexity\n{self.label_map[y_param]}')
        ax.grid(alpha=0.6)

        # open and slice exemplary meshes
        for i, id_ in enumerate(ids):

            mesh = trimesh.load_mesh(
                fr'../combinations/{id_}_discontinuities.stl')
            section = mesh.section(plane_origin=[5, 5, 5],
                                   plane_normal=[0, 0, 1])
            # project section to 2D with explicit transformation matrix
            section_2D, matrix = section.to_2D(
                to_2D=np.array([[1., 0., 0., -5.],
                                [0., 1., 0., -5.],
                                [0., 0., 1., -5.],
                                [0., 0., 0., 1]]))

            ax = fig.add_subplot(gs[1, i])

            for entity in section_2D.entities:
                # if the entity has it's own plot method use it
                if hasattr(entity, 'plot'):
                    entity.plot(section_2D.vertices)
                    continue
                # otherwise plot the discrete curve
                discrete = entity.discrete(section_2D.vertices)
                ax.plot(*discrete.T, color='dimgrey', lw=.2)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(r'../output/graphics/complexity_scatter.png', dpi=1200)

        if save_high_density is True:
            fig_hd, ax_hd = plt.subplots(figsize=(10, 10))
            for entity in section_2D.entities:
                # if the entity has it's own plot method use it
                if hasattr(entity, 'plot'):
                    entity.plot(section_2D.vertices)
                    continue
                # otherwise plot the discrete curve
                discrete = entity.discrete(section_2D.vertices)
                ax_hd.plot(*discrete.T, color='dimgrey', lw=.5)
            ax_hd.set_xlim(-5, 5)
            ax_hd.set_ylim(-5, 5)
            ax_hd.set_xticks([])
            ax_hd.set_yticks([])
            ax_hd.set_aspect('equal')
            plt.tight_layout()
            plt.savefig(
                r'../output/graphics/complexity_scatter_sample.svg')

    def advanced_parameter_scatter_plot(self, df: pd.DataFrame,
                                        close: bool = True) -> None:
        '''function plots structural complexity against different other
        parameters'''
        def add_scatter(ax: plt.axis, x: str, y: str,
                        yscale: str = None) -> None:
            ax.scatter(df[x], df[y], alpha=0.5, color='black',
                       edgecolor='grey', s=10)
            if yscale == 'log':
                ax.set_yscale('log')
            ax.set_xlabel(self.label_map[x])
            ax.set_ylabel(self.label_map[y])
            ax.grid(alpha=0.5)

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6.85039, 9.2126))

        add_scatter(axs[0, 0], 'P32', 'Shannon entropy')
        add_scatter(axs[0, 1], 'P32', 'compression ratio')
        add_scatter(axs[1, 0], 'P32', 'structural complexity')
        add_scatter(axs[1, 1], 'P32', 'Euler characteristic')
        add_scatter(axs[2, 0], 'P32', 'Minkowski dimension')
        add_scatter(axs[2, 1], 'P32', 'Euler characteristic inverted')

        plt.tight_layout()
        # subplot labels
        x1, x2 = 0.13, 0.66
        plt.text(x1, 0.96, 'a)', fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.text(x2, 0.96, 'b)', fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.text(x1, 0.63, 'c)', fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.text(x2, 0.63, 'd)', fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.text(x1, 0.30, 'e)', fontsize=14,
                 transform=plt.gcf().transFigure)
        plt.text(x2, 0.30, 'f)', fontsize=14,
                 transform=plt.gcf().transFigure)

        plt.savefig(r'../output/graphics/advanced_parameter_plot.svg')
        plt.savefig(r'../output/graphics/advanced_parameter_plot.pdf')
        if close is True:
            plt.close()

    def complexity_scatter_1(self, df: pd.DataFrame, s: int = 3,
                             fsize: float = 8) -> None:
        '''plot that scatters the complexity parameters that increase and then
        decrease with discontinuity density'''

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 5),
                                layout='constrained')
        ax2 = axs[0].twinx()
        ax3 = axs[0].twinx()

        axs[0].set_ylabel(self.label_map['compression ratio'], fontsize=fsize)
        ax2.set_ylabel(self.label_map['Shannon entropy'], fontsize=fsize)
        ax3.set_ylabel(self.label_map['structural complexity'], fontsize=fsize)
        axs[1].set_ylabel(self.label_map['Euler characteristic'], fontsize=fsize)
        axs[1].set_xlabel(self.label_map['P32'], fontsize=fsize)

        axs[0].scatter(df['P32'], df['compression ratio'], s=s, color='black',
                       alpha=0.2, label=self.label_map['compression ratio'])
        ax2.scatter(df['P32'], df['Shannon entropy'], s=s, color='C0',
                    alpha=0.2, label=self.label_map['Shannon entropy'])
        ax3.scatter(df['P32'], df['structural complexity'], s=s, color='C1',
                    alpha=0.2, label=self.label_map['structural complexity'])
        axs[1].scatter(df['P32'], df['Euler characteristic'], s=s,
                       color='black', alpha=0.2,
                       label=self.label_map['Euler characteristic'])

        axs[0].grid(alpha=0.5)
        axs[1].grid(alpha=0.5)
        # move third y axis further to the right
        ax3.spines['right'].set_position(('outward', 40))

        axs[0].tick_params(axis='both', labelsize=fsize)
        axs[0].xaxis.set_ticklabels([])
        ax2.tick_params(axis='both', labelsize=fsize)
        ax3.tick_params(axis='both', labelsize=fsize)
        axs[1].tick_params(axis='both', labelsize=fsize)

        axs[0].yaxis.label.set_color('black')
        ax2.yaxis.label.set_color('C0')
        ax3.yaxis.label.set_color('C1')

        # subplot annotation
        plt.gcf().text(0.02, 0.97, 'a)', fontsize=fsize+4)
        plt.gcf().text(0.02, 0.47, 'b)', fontsize=fsize+4)

        plt.savefig(r'../output/graphics/complexity_scatter_1.svg')
        plt.savefig(r'../output/graphics/complexity_scatter_1.pdf')
        plt.close()

    def complexity_scatter_2(self, df: pd.DataFrame, s: int = 3,
                             fsize: float = 8) -> None:
        '''plot that scatters the complexity parameters that only increase with
        discontinuity density'''

        fig, ax1 = plt.subplots(figsize=(4, 3.5), layout='constrained')
        ax2 = ax1.twinx()

        ax1.set_xlabel(self.label_map['P32'], fontsize=fsize)
        ax1.set_ylabel(self.label_map['Minkowski dimension'], fontsize=fsize)
        ax2.set_ylabel(self.label_map['Euler characteristic inverted'],
                       fontsize=fsize)

        p1 = ax1.scatter(df['P32'], df['Minkowski dimension'], s=s,
                         color='black', alpha=0.2,
                         label=self.label_map['Minkowski dimension'])
        p2 = ax2.scatter(df['P32'], df['Euler characteristic inverted'], s=s,
                         color='C1', alpha=0.2,
                         label=self.label_map['Euler characteristic inverted'])

        ax2.legend(handles=[p1, p2], loc='lower right', framealpha=1,
                   fontsize=fsize)
        ax1.grid(alpha=0.5)

        ax1.tick_params(axis='both', labelsize=fsize)
        ax2.tick_params(axis='both', labelsize=fsize)

        ax1.yaxis.label.set_color('black')
        ax2.yaxis.label.set_color('C1')

        # plt.tight_layout()
        plt.savefig(r'../output/graphics/complexity_scatter_2.svg')
        plt.savefig(r'../output/graphics/complexity_scatter_2.pdf')
        plt.close()

    def pairplot(self, df: pd.DataFrame, plot_params: list,
                 fsize: float = 7) -> None:
        '''pairplot'''
        n_params = len(plot_params)
        counter = 1

        fig = plt.figure(figsize=(16, 16))
        for i in range(n_params):
            for j in range(n_params):
                ax = fig.add_subplot(n_params, n_params, counter)
                if i == j:  # diagonal
                    ax.hist(df[plot_params[i]], bins=30, color='grey',
                            edgecolor='black')
                    ax.set_xlabel(plot_params[i], fontsize=fsize)
                else:  # scatter
                    ax.scatter(df[plot_params[i]], df[plot_params[j]],
                               color='grey', edgecolor='black', s=2, alpha=0.3)
                    ax.set_xlabel(self.label_map[plot_params[i]],
                                  fontsize=fsize)
                    ax.set_ylabel(self.label_map[plot_params[j]],
                                  fontsize=fsize)
                ax.tick_params(axis='both', labelsize=fsize)
                counter += 1

        plt.tight_layout()
        plt.savefig(r'../output/graphics/pairplot_general.svg')
        plt.savefig(r'../output/graphics/pairplot_general.pdf')
        plt.savefig(r'../output/graphics/pairplot_general.jpg', dpi=600)
        plt.close()

    def Jv_plot(self, df: pd.DataFrame, Jv_s: list,
                limit: float = 100) -> None:
        '''figure that scatters different versions of the computed volumetric
        joint count against each other'''
        fig = plt.figure(figsize=(6, 6))

        ax = fig.add_subplot(1, 1, 1)
        markers = ['o', 'v', 'P', 's', 'X', 'D'] * 2
        for i, jv in enumerate(Jv_s):
            x, y = 'Jv measured [discs/m³]', jv
            df_temp = df.dropna(subset=[x, y])
            r2 = r2_score(df_temp[x], df_temp[y])
            if r2 < 0:
                r2 = '< 0'
            else:
                r2 = round(r2, 2)
            ax.scatter(df_temp[x], df_temp[y], alpha=0.5,
                       label=f'{jv}; R2: {r2}', marker=markers[i])

        ax.set_xlim(left=0, right=limit)
        ax.set_ylim(bottom=0, top=limit)
        ax.grid(alpha=0.5)
        ax.set_xlabel('Jv measured [disc./m³]')
        ax.set_ylabel('Jv computed [disc./m³]')
        ax.plot([0, limit], [0, limit], color='black')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(r'../output/graphics/JVs.pdf', dpi=600)
        plt.close()

    def relation_type_pairplot(self, df: pd.DataFrame, plot_params: list,
                               relation_dic: dict, fsize: float = 7) -> None:
        '''pairplot that shows scatterplots of parameters against each other
        above the diagonal, histograms in the diagonal and relationship types
        below the diagonal'''
        n_params = len(plot_params)
        counter = 1

        fig = plt.figure(figsize=(16, 16))
        for i in range(n_params):
            for j in range(n_params):
                ax = fig.add_subplot(n_params, n_params, counter)
                if i == j:  # diagonal
                    ax.hist(df[plot_params[i]], bins=30, color='grey',
                            edgecolor='black')
                    ax.set_xlabel(plot_params[i], fontsize=fsize)
                if i < j:  # scatter
                    ax.scatter(df[plot_params[i]], df[plot_params[j]],
                               color='grey', edgecolor='black', s=1, alpha=0.3)
                    ax.set_xlabel(plot_params[i], fontsize=fsize)
                    ax.set_ylabel(plot_params[j], fontsize=fsize)
                else:  # relation type
                    if [plot_params[i], plot_params[j]] in relation_dic['linear']:
                        ax.text(x=0.5, y=0.5, s='li', fontsize=fsize*3,
                                ha='center', va='center')
                        ax.set_facecolor('goldenrod')
                    elif [plot_params[i], plot_params[j]] in relation_dic['exponential']:
                        ax.text(x=0.5, y=0.5, s='ex', fontsize=fsize*3,
                                ha='center', va='center')
                        ax.set_facecolor('lightgrey')
                    elif [plot_params[i], plot_params[j]] in relation_dic['powerlaw']:
                        ax.text(x=0.5, y=0.5, s='po', fontsize=fsize*3,
                                ha='center', va='center')
                        ax.set_facecolor('orangered')
                    ax.set_xlabel(plot_params[i], fontsize=fsize)
                    ax.set_ylabel(plot_params[j], fontsize=fsize)
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.tick_params(axis='both', labelsize=fsize)
                counter += 1

        plt.tight_layout()
        plt.savefig(r'../output/graphics/pairplot_relation.pdf', dpi=600)
        plt.close()

    def scatter_combinations(self, df: pd.DataFrame, relation_dic: dict,
                             plot_params: list) -> None:
        '''function creates scatter plots of one parameter against another'''
        n_params = len(plot_params)
        params_dict = dict(zip(plot_params, list(range(len(plot_params)))))

        for i in range(n_params):
            for j in range(n_params):
                if i != j and j > i:
                    x, y = plot_params[i], plot_params[j]
                    # fit function to data
                    df_tmp = df.dropna(subset=[x, y])
                    df_tmp = df_tmp.sort_values(by=x)

                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(df_tmp[x], df_tmp[y], alpha=0.5)
                    # in case of linear best fit
                    if [x, y] in relation_dic['linear']:
                        z_1d = np.polyfit(df_tmp[x].values, df_tmp[y].values,
                                          1)
                        p_1d = np.poly1d(z_1d)
                        function_vals = p_1d(df_tmp[x].values)
                        r2 = round(r2_score(df_tmp[y].values, function_vals),
                                   2)
                        ax.plot(df_tmp[x].values, function_vals, color='black',
                                label=f'linear fit, r2: {r2}\n{y} = {round(z_1d[0], 6)} * {x} + {round(z_1d[1], 6)}')

                    elif [x, y] in relation_dic['exponential']:
                        a, b, c = self.fit_exponential(df_tmp[x].values,
                                                       df_tmp[y].values)
                        y_fit = self.exponential(df_tmp[x].values, a, b, c)
                        r2 = round(r2_score(df_tmp[y].values, y_fit), 2)
                        ax.plot(df_tmp[x].values, y_fit, color='black',
                                label=f'exponential fit, r2: {r2}\n{a, b, c}')

                    elif [x, y] in relation_dic['powerlaw']:
                        if x == 'avg. RQD' and y == 'Minkowski dimension':
                            a, b, c = self.fit_power_law_neg(df_tmp[x].values,
                                                      df_tmp[y].values)
                            y_fit = self.power_law_neg(df_tmp[x].values, a, b,
                                                       c)
                            r2 = round(r2_score(df_tmp[y].values, y_fit), 2)
                            ax.plot(df_tmp[x].values, y_fit, color='black',
                                    label=f'powerlaw fit, r2: {r2}\n{y} = {a}*{x}^{round(b, 5)}+{round(c, 5)}')
                        else:
                            a, b = self.fit_power_law(df_tmp[x].values,
                                                      df_tmp[y].values)
                            y_fit = self.power_law(df_tmp[x].values, a, b)
                            r2 = round(r2_score(df_tmp[y].values, y_fit), 2)
                            ax.plot(df_tmp[x].values, y_fit, color='black',
                                    label=f'powerlaw fit, r2: {r2}\n{y} = {round(a, 6)}*{x}^{round(b, 6)}')

                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.grid(alpha=0.5)
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(fr'../output/graphics/scatters/{params_dict[x]}_{params_dict[y]}.svg', dpi=150)
                    plt.close()

    def RQD_spacing_hist_plot(self, df: pd.DataFrame) -> None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 7))

        ax1.hist(df['avg. RQD'], color='grey', edgecolor='black', bins=30)
        ax1.set_title('avg. RQD')

        ax2.hist(df['avg. app. spacing [m]'], color='grey', edgecolor='black',
                 bins=30)
        ax2.set_title('avg. app. spacing [m]')

        plt.tight_layout()
        plt.savefig(r'../output/graphics/RQD_hist.pdf', dpi=600)
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

        fig.text(x=0.55, y=0.96, s='dimension of measurement', ha='center',
                 fontsize=fsize)
        fig.text(x=0.25, y=0.93, s='0', ha='center', fontsize=fsize)
        fig.text(x=0.55, y=0.93, s='1', ha='center', fontsize=fsize)
        fig.text(x=0.85, y=0.93, s='2', ha='center', fontsize=fsize)

        fig.text(x=0.02, y=0.45, s='dimension of sample', va='center',
                 rotation=90, fontsize=fsize)
        fig.text(x=0.065, y=0.16, s='3D', ha='center', fontsize=fsize)
        fig.text(x=0.065, y=0.45, s='2D', ha='center', fontsize=fsize)
        fig.text(x=0.065, y=0.75, s='1D', ha='center', fontsize=fsize)

        plt.tight_layout(rect=(0.07, 0, 1, 0.93))
        plt.savefig(r'../output/graphics/Pij_plot.pdf', dpi=600)
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
        plt.savefig(r'../output/graphics/directional_plot.pdf', dpi=300)
        plt.close()

    def Q_Jv_plot(self, df: pd.DataFrame) -> None:
        x_min, x_max = 0, 125
        y_min, y_max = 0.1, 100

        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.scatter(df['Jv measured [discs/m³]'], df['Q_struct'],
                         c=df['avg. RQD'], alpha=0.6, vmin=0, vmax=100,
                         zorder=10)
        ax.hlines([1, 4, 10, 40], xmin=x_min, xmax=x_max, color='black',
                  zorder=1)
        ax.vlines([1, 3, 10, 30, 60], ymin=y_min, ymax=y_max, color='black',
                  zorder=1)

        ax.text(1.1, 0.95, 'very poor', va='top')
        ax.text(1.1, 3.9, 'poor', va='top')
        ax.text(1.1, 9.5, 'fair', va='top')
        ax.text(1.1, 39, 'good', va='top')
        ax.text(1.1, 99, 'very good', va='top')

        ax.text(0.95, 0.11, 'very large\nblocks', rotation=-90, ha='right')
        ax.text(2.9, 0.11, 'large blocks', rotation=-90, ha='right')
        ax.text(9, 0.11, 'medium-sized\nblocks', rotation=-90, ha='right')
        ax.text(29, 0.11, 'small blocks', rotation=-90, ha='right')
        ax.text(59, 0.11, 'very small\nblocks', rotation=-90, ha='right')
        ax.text(99, 0.11, 'crushed rock', rotation=-90, ha='right')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([1, 3, 10, 30, 60, 100])
        ax.set_yticks([1, 4, 10, 40, 100])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlim(right=x_max)
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.set_xlabel('Jv measured [discs/m³]')
        ax.set_ylabel('RQD/Jn')

        cbar = plt.colorbar(cax)
        cbar.set_label('RQD')

        plt.tight_layout()
        plt.savefig(r'../output/graphics/Q_Jv_plot.pdf')
        plt.close()

    def Euler_plot(self) -> None:
        '''Figure that plots some generic examples of the Euler
        characteristic'''
        # create generic voxel arrays of shapes
        negative_array = np.ones((5, 5, 5), dtype=int)
        negative_array[1, :, 1] = 0  # Create a tunnel
        negative_array[1, :, 3] = 0  # Create a tunnel
        negative_array[3, :, 1] = 0  # Create a tunnel
        negative_array[3, :, 3] = 0  # Create a tunnel

        intermediate_array = np.ones((5, 5, 5), dtype=int)
        intermediate_array[1, :, :] = 0  # Insert plane
        intermediate_array[3, :, :] = 0  # Insert plane

        positive_array = np.zeros((5, 5, 5), dtype=int)
        positive_array[::2, ::2, ::2] = 1  # Isolated voxels

        # plot
        fig = plt.figure(figsize=(3, 9))
        letters = ['a', 'b', 'c']

        for i, array in enumerate([positive_array, intermediate_array,
                                   negative_array]):
            # array = 1-array
            e = euler_number(array, connectivity=1)  # computer Euler number

            ax = fig.add_subplot(3, 1, i+1, projection='3d')
            ones = np.where(array == 1, True, False)
            zeros = np.where(array == 1, False, True)
            ax.voxels(ones, color='grey', edgecolor='black', lw=0.5,
                      alpha=0.95)
            ax.voxels(zeros, color='white', edgecolor='black', lw=0.5,
                      alpha=0.15)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_axis_off()
            ax.set_title(f'{letters[i]}) χ: {round(e, 2)}')
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(r'../output/graphics/Euler_voxels.svg')
        plt.savefig(r'../output/graphics/Euler_voxels.pdf')
        plt.close()


class parameters:
    '''class with functions that compute rock engineering parameters of rock
    mass structures'''

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
            elif df.iloc[i]['rand_set_ratio'] > 0.25 and len(np.where(df.iloc[i][sets[:3]].values > 0.3)[0]) == 1:
                # 1 set plus random joints
                Jn = 3
            elif len(np.where(df.iloc[i][sets].values > 0.4)[0]) == 1:
                # 1 set
                Jn = 2
            else:
                Jn = 'Jn na'
            Jn_s.append(Jn)

        return Jn_s

    def Minkowski(self, n_boxes: int, box_sizes: float) -> float:
        '''compute the Minkowski–Bouligand dimension as a parameter for fractal
        dimensions.
        https://en.wikipedia.org/wiki/Minkowski%E2%80%93Bouligand_dimension'''
        N_ = np.log(n_boxes)
        eps_ = np.log(1/box_sizes)
        return np.polyfit(eps_, N_, 1)[0]

    def scalar_p_image(self, arr1: np.array, arr2: np.array) -> np.array:
        '''function that computs the scalar product of 2 RGB images'''
        return arr1[:, :, 0] * arr2[:, :, 0] + arr1[:, :, 1] * arr2[:, :, 1] + arr1[:, :, 2] * arr2[:, :, 2]

    def structural_complexity(self, data: np.array, lambda_: int = 2,
                              N: int = None, mode: str = 'image') -> float:
        '''function computes the structural complexity of raster data according
        to Bagrov et al. (2020)
        https://www.pnas.org/doi/10.1073/pnas.2004976117'''
        if N is None:
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
        # print(window_sizes)
        # collection of stack of coarsened raster data
        stack = [data]

        for i, step_size in enumerate(window_sizes):
            if mode == 'image':
                data_c = block_reduce(data, (step_size, step_size, 1),
                                      np.mean)
            elif mode == '3Dgrid':
                data_c = block_reduce(data, (step_size, step_size, step_size),
                                      np.mean)
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

    def Shannon_Entropy(self, df: pd.DataFrame) -> np.array:
        '''compute the information entropy after Shannon 1948 as
        H = -sum(pk * log(pk))
        https://en.wikipedia.org/wiki/Entropy_(information_theory)'''
        counts = df[['n empty voxels at 0.05 [m]',
                     'n disc. voxels at 0.05 [m]']].values
        return entropy(counts, base=2, axis=1)

    def compression_complexity(self, array):
        flattened = array.flatten()
        byte_data = flattened.tobytes()  # Convert to bytes for compression
        compressed_data = zlib.compress(byte_data, level=-1)
        return len(compressed_data) / len(byte_data)

    def lacunarity(self, array: np.array, box_sizes: list,
                   resolution: float) -> dict:
        """
        Computes lacunarity for a 3D binary array over different box sizes.
        Parameters:
        - array: 3D numpy array of binary values (0 and 1)
        - box_sizes: List of integers representing the edge lengths of cubic
        boxes
        - resolution: edge size of boxes
        Returns:
        - lacunarity_results: Dictionary where keys are box sizes and values
        are lacunarity
        """
        lacunarity_results = {}

        for box_size in box_sizes:
            # Apply a sliding window to compute mass within each box
            footprint = np.ones((box_size, box_size, box_size))
            mass = generic_filter(array, np.sum, footprint=footprint,
                                  mode='reflect')
            # Calculate mean and variance of the mass distribution
            mean_mass = np.mean(mass)
            variance_mass = np.var(mass)
            # Compute lacunarity
            if mean_mass > 0:  # Avoid division by zero
                lacunarity = (variance_mass + mean_mass**2) / (mean_mass**2)
            else:
                lacunarity = np.nan  # Undefined if the mean mass is zero
            size = f'{round(box_size*resolution, 2)} m'
            lacunarity_results[size] = lacunarity
        return lacunarity_results
