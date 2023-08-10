# -*- coding: utf-8 -*-
"""
Code to the paper "Rock mass structure characterization considering finite and
folded discontinuities"
Dr. Georg H. Erharter - 2023
DOI: XXXXXXXXXXX

Script that generates specific plots for the publication.
"""

# defining the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


from X_library import utilities

utils = utilities()


def RQD_Jv_Palmstrøm_1974(Jv):
    if Jv < 4.5:
        RQD = 100
    elif Jv >= 35:
        RQD = 0
    else:
        RQD = 115 - 3.3 * Jv
    return RQD


def RQD_Jv_Palmstrøm_2005(Jv):
    if Jv < 4:
        RQD = 100
    elif Jv >= 44:
        RQD = 0
    else:
        RQD = 110 - 2.5 * Jv
    return RQD


df = pd.read_excel(r'../output/PDD1_1.xlsx')

################################################### RQD - D
df.dropna(subset=['Minkowski dimension'], inplace=True)
a, b, c = -2.1e-7, 3.3, 2.75
RQD_s = np.linspace(0, 100, 100)
D_s = utils.power_law_neg(RQD_s, a, b, c)
y_pred = utils.power_law_neg(df['avg. RQD'], a, b, c)
r2 = round(r2_score(df['Minkowski dimension'].values, y_pred), 2)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(df['avg. RQD'], df['Minkowski dimension'], alpha=0.5)
ax.plot(RQD_s, D_s, color='black',
        label=f'powerlaw fit, r2: {r2}\nD = {a}*RQD^{round(b, 5)}+{round(c, 5)}')
ax.grid(alpha=0.5)
ax.set_xlabel('avg. RQD')
ax.set_ylabel('Minkowski dimension')
ax.legend()

plt.tight_layout()
plt.savefig(r'../graphics/scatters/5_6_man.png', dpi=150)
plt.close()

################################################### P10
a, b, c = 100, 0.1, 1
p10_s = np.linspace(0, 50, 100)
RQD_p10 = utils.exponential(p10_s, a, b, c)
y_pred = utils.exponential(df['avg. P10'], a, b, c)
r2_RQD_P10 = round(r2_score(df['avg. RQD'].values, y_pred), 2)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(df['avg. P10'], df['avg. RQD'], alpha=0.5)
ax.plot(p10_s, RQD_p10, color='black',
        label=f'exponential fit, r2: {r2_RQD_P10}\nRQD = {a}*e^(-{b}*P10) ({b} * P10 + {c})')
ax.grid(alpha=0.5)
ax.set_xlabel('avg. P10')
ax.set_ylabel('avg. RQD')
ax.legend()

plt.tight_layout()
plt.savefig(r'../graphics/scatters/0_5_man.png', dpi=150)
plt.close()

################################################### P20
a, b, c = 100, 0.38, 1
p20_s = np.linspace(0, 10, 100)
RQD = utils.exponential(p20_s, a, b, c)
y_pred = utils.exponential(df['avg. P20'], a, b, c)
r2 = round(r2_score(df['avg. RQD'].values, y_pred), 2)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(df['avg. P20'], df['avg. RQD'], alpha=0.5)
ax.plot(p20_s, RQD, color='black',
        label=f'exponential fit, r2: {r2}\nRQD = {a}*e^(-{b}*P20) ({b} * P20 + {c})')
ax.grid(alpha=0.5)
ax.set_xlabel('avg. P20')
ax.set_ylabel('avg. RQD')
ax.legend()

plt.tight_layout()
plt.savefig(r'../graphics/scatters/1_5_man.png', dpi=150)
plt.close()

################################################### P21
a, b, c = 100, 0.07, 1
p21_s = np.linspace(0, 70, 100)
RQD = utils.exponential(p21_s, a, b, c)
y_pred = utils.exponential(df['avg. P21'], a, b, c)
r2 = round(r2_score(df['avg. RQD'].values, y_pred), 2)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(df['avg. P21'], df['avg. RQD'], alpha=0.5)
ax.plot(p21_s, RQD, color='black',
        label=f'exponential fit, r2: {r2}\nRQD = {a}*e^(-{b}*P21) ({b} * P21 + {c})')
ax.grid(alpha=0.5)
ax.set_xlabel('avg. P21')
ax.set_ylabel('avg. RQD')
ax.legend()

plt.tight_layout()
plt.savefig(r'../graphics/scatters/2_5_man.png', dpi=150)
plt.close()

################################################### P32
a, b, c = 100, 0.06, 1
p32_s = np.linspace(0, 80, 100)
RQD = utils.exponential(p32_s, a, b, c)
y_pred = utils.exponential(df['P32'], a, b, c)
r2 = round(r2_score(df['avg. RQD'].values, y_pred), 2)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(df['P32'], df['avg. RQD'], alpha=0.5)
ax.plot(p32_s, RQD, color='black',
        label=f'exponential fit, r2: {r2}\nRQD = {a}*e^(-{b}*P32) ({b} * P32 + {c})')
ax.grid(alpha=0.5)
ax.set_xlabel('P32')
ax.set_ylabel('avg. RQD')
ax.legend()

plt.tight_layout()
plt.savefig(r'../graphics/scatters/3_5_man.png', dpi=150)
plt.close()

################################################### Jv
a, b, c = 100, 0.037, 1
Jv_s = np.linspace(0, 120, 100)
RQD_erh = utils.exponential(Jv_s, a, b, c)
RQD_Palm_1974 = [RQD_Jv_Palmstrøm_1974(jv) for jv in Jv_s]
RQD_Palm_2005 = [RQD_Jv_Palmstrøm_2005(jv) for jv in Jv_s]

y_pred = utils.exponential(df['Jv measured [discs/m³]'], a, b, c)
r2_RQD_Jv = round(r2_score(df['avg. RQD'].values, y_pred), 2)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(df['Jv measured [discs/m³]'], df['avg. RQD'], alpha=0.5)
ax.plot(Jv_s, RQD_erh, color='black',
        label=f'exponential fit, r2: {r2_RQD_Jv}\nRQD = {a}*e^(-{b}*Jv) ({b} * Jv + {c})')
ax.plot(Jv_s, RQD_Palm_1974, color='black', ls='--',
        label='Palmstrøm 1974')
ax.plot(Jv_s, RQD_Palm_2005, color='black', ls='-.',
        label='Palmstrøm 2005')
ax.grid(alpha=0.5)
ax.set_xlabel('Jv measured [discs/m³]')
ax.set_ylabel('avg. RQD')
ax.legend()

plt.tight_layout()
plt.savefig(r'../graphics/scatters/4_5_man.png', dpi=150)
plt.close()

###################################################
# start figure for paper where two relationships are plotted
fig_m, (ax1_m, ax2_m) = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 10.5))

ax1_m.scatter(df['avg. P10'], df['avg. RQD'], color='grey', edgecolor='black',
              alpha=0.1)
ax1_m.plot(p10_s, RQD_p10, color='black', lw=3,
           label=f'R2: {r2_RQD_P10}\nRQD = 100*e^(-0.1*P10) (0.1*P10+1)\n(Priest and Hudson, 1976)')
ax1_m.grid(alpha=0.5)
ax1_m.set_xlabel('P10 [n discontinuities / meter]')
ax1_m.set_ylabel('RQD')
ax1_m.legend()

ax2_m.scatter(df['Jv measured [discs/m³]'], df['avg. RQD'], color='grey',
              edgecolor='black', alpha=0.1)
ax2_m.plot(Jv_s, RQD_erh, color='black', lw=3,
           label=f'R2: {r2_RQD_Jv}\nRQD = 100*e^(-0.037*Jv) (0.037*RQD+1)')
ax2_m.plot(Jv_s, RQD_Palm_1974, color='grey', ls='-', lw=3,
           label='Palmstrøm (1974)')
ax2_m.plot(Jv_s, RQD_Palm_2005, color='black', ls=':', lw=3,
           label='Palmstrøm (2005)')
ax2_m.grid(alpha=0.5)
ax2_m.set_xlabel('Jv measured [discs/m³]')
ax2_m.set_ylabel('RQD')
ax2_m.legend(loc='upper right')

plt.tight_layout()
plt.savefig(r'../graphics/relationships_selected.svg', dpi=600)
plt.close()

