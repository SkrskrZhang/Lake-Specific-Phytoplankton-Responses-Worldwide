import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib.patheffects import Normal
from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from matplotlib.colors import LogNorm
import dtreeviz
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.tree import plot_tree
from matplotlib.patches import Wedge, Patch
from collections import defaultdict
import plotly.graph_objects as go
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
import chardet
from statsmodels.genmod.families import Family, Gaussian, Poisson, Binomial, Gamma, NegativeBinomial
from statsmodels.genmod.families.links import identity, log, logit, inverse_power
import warnings
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from mapie.regression import SplitConformalRegressor
from sklearn.model_selection import train_test_split
from collections import Counter
import pymc as pm
import arviz as az
# from dominance_analysis import Dominance
import itertools
import geopandas as gpd
import math
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
from xlrd import okind_dict
import shap
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

import random
random.seed(911)
np.random.seed(911)

from merge_data import *
from config import *
from uilts import *
from figs import fig2


def fig2_global_wq_map_responses_and_n2p_ratio():
    data_df, meta_df = get_lake_data(easy_get=True, wq_cols=['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'], min_wq=0.01, max_wq=100000)
    meta_df = meta_df.dropna(subset=['Ch_mean', 'TP_mean', 'TN_mean', 'Latitude', 'Longitude'])
    print('used lakes', len(meta_df))
    data_df = data_df[data_df['TP (µg/L)'] > 1]
    data_df = data_df[data_df['TN (µg/L)'] > 10]
    data_df = data_df[data_df['Chla (µg/L)'] > 0.1]


    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    m = Basemap(projection='cyl', resolution='i', ax=axes[0])
    m.drawcountries(linewidth=0.1)
    m.fillcontinents(color='lightgray', lake_color='lightgray', alpha=.5)
    m.drawmapboundary(fill_color='none', linewidth=0.5)
    parallels = np.arange(-80, 81, 20)
    meridians = np.arange(-180, 181, 60)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.25, dashes=[5, 0], fontsize=8, color='lightgray')
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.25, dashes=[5, 0], fontsize=8, color='lightgray')

    x, y = m(meta_df['Longitude'].values, meta_df['Latitude'].values)
    values = meta_df['Ch_mean'].values
    s = m.scatter(
        x, y,
        c=values,
        cmap='jet',
        norm=LogNorm(),
        marker='o',
        s=7,
        edgecolor='k',
        linewidths=0.2,
        alpha=1,
    )
    cb = m.colorbar(s, location='right', size="2%", pad='2%')
    cb.set_label("Lake mean Chl$a$ (µg/L)", fontsize=8)
    cb.ax.set_yscale('log')
    cb.ax.tick_params(labelsize=9, width=.5, length=3)
    cb.outline.set_linewidth(.5)
    axes[0].set_aspect('auto')


    y_col = 'Chla (µg/L)'
    for i, x_col in enumerate(['TP (µg/L)', 'TN (µg/L)']):
        plt.sca(axes[i+1])
        plt.scatter(data_df[x_col].values, data_df[y_col].values, alpha=0.5, c='#A4A4A4', edgecolors='k', s=7, linewidths=.25)
        lowess_result = sm.nonparametric.lowess(data_df[y_col].values, data_df[x_col].values, frac=0.2)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c='#1C2833', ls='-', linewidth=2)
        slope, intercept, r_value, p_value, std_err = linregress(np.log10(data_df[x_col].values), np.log10(data_df[y_col].values))
        print(x_col, r_value)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(x_col, fontsize=9)
        plt.ylabel(y_col, fontsize=9)

    for ax, label in {axes[0]: 'a', axes[1]: 'b', axes[2]: 'c'}.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('Figs/SI 2 Global Lake water quality map.png', dpi=300)


    plt.figure(figsize=(8, 6))
    data_df['TP_log10'] = np.log10(data_df['TP (µg/L)'].values)
    data_df['TN2TP'] = data_df['TN (µg/L)'].values / data_df['TP (µg/L)'].values
    data_df['TN_std'] = calc_std_tn(data_df['TP (µg/L)'])
    data_df['TN2TP_std'] = data_df['TN_std'].values / data_df['TP (µg/L)'].values
    data_df['ln_t'] = np.where(data_df['TN2TP'].values > data_df['TN2TP_std'].values, 1, 0)
    print('P limit perc', data_df['ln_t'].mean())

    plt.scatter(data_df['TP (µg/L)'].values, data_df['TN2TP'].values, alpha=0.5, c='#A4A4A4', edgecolors='k', s=7, linewidths=.25)
    tp_samples = np.linspace(1, 5000, 3000)
    tn_samples = calc_std_tn(tp_samples)
    plt.plot(tp_samples, tn_samples/tp_samples, c='r', lw=2, label='Critical TN:TP')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('TP (µg/L)')
    plt.ylabel('TN:TP')
    plt.yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
    plt.gca().legend(loc='lower left')
    plt.gca().grid(True, which='both', axis='x')

    ax2_twin = plt.gca().twinx()
    tp_log_slice = np.linspace(data_df['TP_log10'].min(), data_df['TP_log10'].max(), 50)
    tp_mean_ls, ratio_ls = [], []
    for i in range(len(tp_log_slice)-1):
        sub_df = data_df[(data_df['TP_log10'] >= tp_log_slice[i]) & (data_df['TP_log10'] <= tp_log_slice[i+1])]
        tp_mean_ls.append(sub_df['TP (µg/L)'].median())
        ratio_ls.append(sub_df['ln_t'].mean())
    line1, = ax2_twin.plot(tp_mean_ls, ratio_ls, lw=2, label='Proportion of P limitation')
    ax2_twin.set_yticks([-.2, 0, .2, .4, .6, .8, 1, 1.2], ['', "0", "20", "40", "60", "80", "100", ''])
    ax2_twin.set_ylabel('Proportion of P limitation (%)')
    ax2_twin.legend([line1], ['Proportion of P limitation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('Figs/SI 2_2 Global Lake N2P.png', dpi=300)
    plt.show()


def fig3_lake_wq_record_number():
    tp_data_df, tp_meta_df = get_lake_data(easy_get=True, wq_cols=['Chla (µg/L)', 'TP (µg/L)'], min_wq=1, max_wq=200)
    tn_data_df, tn_meta_df = get_lake_data(easy_get=True, wq_cols=['Chla (µg/L)', 'TN (µg/L)'], min_wq=1, max_wq=6000)
    tnp_data_df, tnp_meta_df = get_lake_data(easy_get=True, wq_cols=['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'], min_wq=1, max_wq=6000)

    meta_dic = {'tp': tp_meta_df, 'tn': tn_meta_df, 'tnp': tnp_meta_df}
    record_ranges = range(1, 101)
    n_lake_df = pd.DataFrame(index=record_ranges, columns=list(meta_dic.keys()))
    n_lake_df.loc[:, :] = 0
    for col, df in meta_dic.items():
        for record in record_ranges:
            filter_df = df[(df['n_samples'] >= record) & (df['n_month'] >= 6)]
            if len(filter_df) > 0:
                n_lake_df.loc[record, col] = filter_df.shape[0]

    xlabel_ls = ['Minimum number of records\n(Concurrent Chla & TP)', 'Minimum number of records\n(Concurrent Chla & TN)', 'Minimum number of records\n(Concurrent Chla & TP & TN)']
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey='row')
    for i, col in enumerate(['tp', 'tn', 'tnp']):
        plt.sca(axes[i])
        plt.bar(record_ranges, n_lake_df[col].values, width=1, color=fill_c, edgecolor=line_c, linewidth=0.2, alpha=0.7)
        plt.xlabel(xlabel_ls[i])
        plt.axvline(30, color=mark_c, linestyle='--', linewidth=1, alpha=0.9)
        if i == 0:
            plt.ylabel('Number of lakes')
        plt.xlim(0, 100)

    for ax, label in {axes[0]: 'a', axes[1]: 'b', axes[2]: 'c'}.items():
        ax.text(0.95, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

    plt.tight_layout()
    plt.savefig('Figs/SI 3 Lake water quality records.png', dpi=300)
    plt.show()


def fig34_lake_tp_chl_box():
    data_df, meta_df = get_lake_data(easy_get=True, wq_cols=['Chla (µg/L)', 'TP (µg/L)'])

    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]
    print('used lakes', len(meta_df))
    data_df = data_df[data_df['Unique Lake'].isin(meta_df.index)]

    lake_median_tp = data_df.groupby('Unique Lake')['TP (µg/L)'].median()
    sorted_lakes = lake_median_tp.sort_values().index.values
    TP_list = []
    Chla_list = []
    for lake in sorted_lakes:
        sub = data_df[data_df['Unique Lake'] == lake]
        TP_list.append(sub['TP (µg/L)'].values)
        Chla_list.append(sub['Chla (µg/L)'].values)
        print(lake)


    fig, axes = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
    boxprops = dict(
        linewidth=0.5,
        color='k'
    )

    whiskerprops = dict(
        linewidth=0.25,
        color='grey'
    )

    capprops = dict(
        linewidth=0.2,
        color='k'
    )

    medianprops = dict(
        linewidth=2,
        color='r'
    )

    flierprops = dict(
        marker='.',
        markersize=0.1,
        markerfacecolor='0.8',
        markeredgecolor='0.8',
        alpha=0.6
    )

    for i, ax in enumerate(axes):
        plt.sca(axes[i])
        if i == 0:
            ls = TP_list
            plt.ylabel("TP (µg/L)")
        else:
            ls = Chla_list
            plt.ylabel("Chla (µg/L)")
        plt.boxplot(
            ls,
            widths=0.9,
            showfliers=True,
            showcaps=False,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops,
            flierprops=flierprops
        )

        if i == 0:
            plt.ylim(1, 1000)
        if i == 1:
            plt.ylim(1, 300)
        if i == 1:
            plt.xlabel("577 lakes (sorted by median TP)")
        plt.yscale('log')
        plt.xticks([])
    plt.tight_layout()
    plt.savefig('Figs/SI 3to4 Lake water quality records.png', dpi=300)


def fig4_var_pair_plot():
    data_df, meta_df = get_lake_data(easy_get=True, wq_cols=['Chla (µg/L)', 'TP (µg/L)'])

    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]
    print('used lakes', len(meta_df))

    data_df = data_df[data_df['Unique Lake'].isin(meta_df.index)]
    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)

    meta_df.loc[mean_df.index, 'Ch_mean'] = mean_df['Chla (µg/L)'].values
    meta_df.loc[mean_df.index, 'TP_mean'] = mean_df['TP (µg/L)'].values
    print('used lakes', len(meta_df))


    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Latitude'] = np.abs(meta_df['Latitude'])
    meta_df['Altitude (m)'] = np.clip(meta_df['Altitude (m)'], 1, None)
    cols = ['Latitude', 'Altitude (m)',
            'temp_mean', 'sr_mean', 'pr_mean', 'ws_mean',
            'Depth_avg', 'Lake_area', 'Res_time', 'Shore_dev', 'Wind_exposure_idx',
            'TP_mean', 'Ch_mean'
            ]

    meta_df = meta_df.dropna(subset=cols)
    print('used lakes', len(meta_df))


    for col in cols[1:]:
        meta_df[col] = np.log10(meta_df[col])
    meta_df = meta_df.dropna(subset=cols)
    print('used lakes', len(meta_df))
    print(meta_df.columns)

    def text_r(x, y, color):
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        x_samples = np.linspace(x.min(), x.max(), 100)
        y_pred = slope * x_samples + intercept

        lowess_result = sm.nonparametric.lowess(y, x, frac=0.3)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=1, alpha=0.6)

        if abs(r_value) > 0.3 and p_value < 0.05:
            plt.text(0.02, 1, '{:.2f}'.format(r_value) if p_value < 0.05 else '',
                     transform=plt.gca().transAxes, fontsize=7, color=color, va='top', ha='left', fontweight='bold')
        else:
            plt.text(0.02, 1, '{:.2f}'.format(r_value) if p_value < 0.05 else '',
                     transform=plt.gca().transAxes, fontsize=7, color='k', va='top', ha='left', fontweight='bold', alpha=0.7)

    abb_dic = {
        'Lat': 'Latitude',
        'log(Alt)': 'Altitude (m)',

        'log(T)': 'temp_mean',
        'log(SR)': 'sr_mean',
        'log(WS)': 'ws_mean',
        'log(Pr)': 'pr_mean',

        'log(Depth)': 'Depth_avg',
        'log(Area)': 'Lake_area',
        'log(RT)': 'Res_time',
        'log(SDI)': 'Shore_dev',
        'log(WEI)': 'Wind_exposure_idx',

        'log(TP)': 'TP_mean',
        'log(Chla)': 'Ch_mean',
    }
    for k, v in abb_dic.items():
        meta_df[k] = meta_df[v]

    new_cols = list(abb_dic.keys())

    g = sns.PairGrid(meta_df[new_cols], corner=True, diag_sharey=False)
    g.map_lower(
        sns.scatterplot,
        s=5,
        alpha=0.4,
        color=scat_c,
        edgecolor=line_c
    )
    g.map_lower(
        text_r, color=mark_c
    )
    g.map_diag(
        sns.kdeplot,
        fill=True,
        color=fill_c,
        alpha=0.75,
        linewidth=0
    )
    max_vals = []
    for col in new_cols:
        from scipy.stats import gaussian_kde
        data = meta_df[col].dropna().values
        kde = gaussian_kde(data)
        x_eval = np.linspace(data.min(), data.max(), 1000)
        y_eval = kde(x_eval)
        max_vals.append(y_eval.max())
    for ax, ymax in zip(g.diag_axes, max_vals):
        ax.set_ylim(0, ymax * 1.25)

    for ax in g.axes.flatten():
        if ax is not None:
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
            ax.tick_params(
                axis='both',
                which='major',
                width=.25,
                length=2,
                labelsize=6,
                direction='in'
            )
            ax.set_xlabel(ax.get_xlabel(), fontsize=6)
            ax.set_ylabel(ax.get_ylabel(), fontsize=6)

    g.fig.subplots_adjust(wspace=0.05, hspace=0.05)
    g.fig.set_size_inches(8, 8)
    plt.tight_layout()
    plt.savefig('Figs/SI 4 Pair plot.png', dpi=300)
    plt.show()


def fig5_model_var_corr():
    data_df, meta_df = get_lake_data(easy_get=True, wq_cols=['Chla (µg/L)', 'TP (µg/L)'])

    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]
    print('used lakes', len(meta_df))

    data_df['log(Chla)'] = np.log10(data_df['Chla (µg/L)'])
    data_df['log(TP)'] = np.log10(data_df['TP (µg/L)'])
    data_df['T'] = data_df['awq_lake_mix_layer_temperature']


    c_dic = {'p': 'log(TP)', 't': 'T', 'c': 'log(Chla)'}

    cols = ['p2c', 't2c', 'p2t']
    r_df = pd.DataFrame(index=meta_df.index, columns=cols)
    for col in cols:
        x_col, y_col = c_dic[col[0]], c_dic[col[2]]
        for lake in meta_df.index:
            lake_df = data_df[data_df['Unique Lake'] == lake]
            x, y = lake_df[x_col].values, lake_df[y_col].values
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            r_df.loc[lake, col] = r_value

    xlabel_ls = ['Intra-lake correlations ($r$)\n(TP vs. Chla)', 'Intra-lake correlations ($r$)\n(T vs. Chla)', 'Intra-lake correlations ($r$)\n(T vs. TP)']
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey='row')
    for i, col in enumerate(cols):
        plt.sca(axes[i])
        plt.hist(r_df[col].values, bins=20, density=True, color=fill_c, alpha=0.7, edgecolor=line_c, linewidth=0.5)

        plt.axvline(0, color=line_c, linestyle='--', linewidth=1, alpha=0.9)
        if i == 0:
            plt.ylabel('Density')
        plt.xlim(-1, 1)
        plt.xlabel(xlabel_ls[i])

    for ax, label in {axes[0]: 'a', axes[1]: 'b', axes[2]: 'c'}.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('Figs/SI 5 model var relation.png', dpi=300)
    plt.show()


def fig6_model_var_sd():
    data_df, meta_df = get_lake_data(easy_get=True, wq_cols=['Chla (µg/L)', 'TP (µg/L)'])

    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]
    print('used lakes', len(meta_df))

    data_df['log(Chla)'] = np.log10(data_df['Chla (µg/L)'])
    data_df['log(TP)'] = np.log10(data_df['TP (µg/L)'])
    data_df['T'] = data_df['awq_lake_mix_layer_temperature']

    xlabel_ls = ['Intra-lake mean Chla (µg/L)', 'Intra-lake mean TP (µg/L)', 'Intra-lake mean T (°C)']
    ylabel_ls = ['Intra-lake CV of Chla', 'Intra-lake CV of TP', 'Intra-lake CV of T']
    x_cols = ['Ch_mean', 'TP_mean', 'temp_mean']
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    cols = ['Ch_cv', 'TP_cv', 'temp_cv']
    for i, col in enumerate(cols):
        plt.sca(axes[i])
        x, y = meta_df[x_cols[i]].values, meta_df[cols[i]].values
        plt.scatter(x, y, alpha=0.5, c=scat_c, edgecolors=line_c, s=20, linewidths=.5)

        lowess_result = sm.nonparametric.lowess(y, x, frac=0.5)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=2)

        plt.ylabel(ylabel_ls[i])
        plt.xlabel(xlabel_ls[i])
        if i < 2:
            plt.xscale('log')

    for ax, label in {axes[0]: 'a', axes[1]: 'b', axes[2]: 'c'}.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('Figs/SI 6 model var sd.png', dpi=300)
    plt.show()


def fig7to9_model_train():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])


    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    with pm.Model() as model:
        alpha_0 = pm.Normal('alpha_0', mu=0, sigma=5)
        beta1_0 = pm.Normal('beta1_0', mu=0, sigma=5)
        beta2_0 = pm.Normal('beta2_0', mu=0, sigma=5)

        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
        sigma_beta1 = pm.HalfNormal('sigma_beta1', sigma=1)
        sigma_beta2 = pm.HalfNormal('sigma_beta2', sigma=1)
        sigma_sigma = pm.HalfNormal('sigma_sigma', sigma=1)

        alpha = pm.Normal('alpha', mu=alpha_0, sigma=sigma_alpha, shape=n_lakes)
        beta1 = pm.Normal('beta1', mu=beta1_0, sigma=sigma_beta1, shape=n_lakes)
        beta2 = pm.Normal('beta2', mu=beta2_0, sigma=sigma_beta2, shape=n_lakes)
        sigma = pm.HalfNormal('sigma', sigma=sigma_sigma, shape=n_lakes)

        mu = alpha[lake_idx] + beta1[lake_idx] * log_TP_centered + beta2[lake_idx] * temp_centered
        chla_obs = pm.LogNormal('chla_obs', mu=mu, sigma=sigma[lake_idx], observed=chla)

        trace = pm.sample(draws=2000, tune=2000, target_accept=0.9, random_seed=911)
        ppc = pm.sample_posterior_predictive(trace)
    summary = az.summary(trace)


    label_dic = {
        'alpha_0': r"$\beta_{0}$",
        'beta1_0': r"$\beta_{1}$",
        'beta2_0': r"$\beta_{2}$",
        'sigma_alpha': r"$\sigma_{\beta_0}$",
        'sigma_beta1': r"$\sigma_{\beta_1}$",
        'sigma_beta2': r"$\sigma_{\beta_2}$",
        'sigma_sigma': r"$\tau$",
        'alpha': r"$\beta_{0,i}$",
        'beta1': r"$\beta_{1,i}$",
        'beta2': r"$\beta_{2,i}$",
        'sigma': r"$\tau_{i}$",
    }


    var_names = ['alpha_0', 'beta1_0', 'beta2_0', 'sigma_alpha', 'sigma_beta1', 'sigma_beta2', 'sigma_sigma']
    fig, axes = plt.subplots(len(var_names), 2, figsize=(8, len(var_names) * 1.5))
    az.plot_trace(trace, var_names=var_names, trace_kwargs={'lw': 0.6, 'alpha': 0.6}, plot_kwargs={'lw': 0.6, 'alpha': 0.6}, axes=axes)
    for i, varname in enumerate(var_names):
        axes[i, 0].tick_params(axis='both', labelsize=8)
        axes[i, 1].tick_params(axis='both', labelsize=8)
        axes[i, 0].set_xlabel('{} distributions in 4 chains'.format(label_dic[varname]), fontsize=9)
        axes[i, 1].set_xlabel('{} samples in 4 chains'.format(label_dic[varname]), fontsize=9)
        if i == 0:
            axes[i, 0].set_title("Posterior distributions", fontsize=10, fontweight='bold')
            axes[i, 1].set_title("MCMC chains", fontsize=10, fontweight='bold')
        else:
            axes[i, 0].set_title("")
            axes[i, 1].set_title("")
    plt.tight_layout()
    plt.savefig('Figs/SI 7 model hyperparameters samples.png', dpi=300)


    az.plot_pair()
    posterior = az.extract(trace, var_names=var_names, combined=True)
    data = {var: posterior[var].values for var in var_names}
    n = len(var_names)
    fig, axes = plt.subplots(n, n, figsize=(9, 8.5))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i < j:
                ax.axis('off')
                continue
            elif i == j:
                ax.hist(data[var_names[i]], bins=20, density=True, color=fill_c, alpha=0.7, edgecolor=line_c, linewidth=0.5)
            else:
                x = data[var_names[j]]
                y = data[var_names[i]]
                print(len(x), len(y))
                ax.scatter(x, y, s=3, alpha=0.3, linewidths=0.25, edgecolors=line_c, color=scat_c)
                r = np.corrcoef(x, y)[0, 1]
                ax.text(0.05, 0.9, f"r = {r:.2f}",
                        transform=ax.transAxes,
                        fontsize=8)
            if i == n - 1:
                ax.set_xlabel(label_dic[var_names[j]], fontsize=8)
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(label_dic[var_names[i]], fontsize=8)
            else:
                ax.set_yticks([])
            ax.tick_params(axis='both', labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
    plt.tight_layout()
    plt.savefig('Figs/SI 7_2 Pair_plot_with_corr.png', dpi=300)


    var_names = ['alpha', 'beta1', 'beta2', 'sigma']
    fig, axes = plt.subplots(len(var_names), 2, figsize=(8, len(var_names) * 1.5))
    az.plot_trace(trace, var_names=var_names, trace_kwargs={'lw': 0.6, 'alpha': 0.6}, plot_kwargs={'lw': 0.6, 'alpha': 0.6}, axes=axes)
    for i, varname in enumerate(var_names):
        axes[i, 0].tick_params(axis='both', labelsize=8)
        axes[i, 1].tick_params(axis='both', labelsize=8)
        axes[i, 0].set_xlabel(label_dic[varname], fontsize=9)
        axes[i, 1].set_xlabel('{} samples in 4 chains'.format(label_dic[varname]), fontsize=9)
        if i == 0:
            axes[i, 0].set_title("Posterior distributions", fontsize=10, fontweight='bold')
            axes[i, 1].set_title("MCMC chains", fontsize=10, fontweight='bold')
        else:
            axes[i, 0].set_title("")
            axes[i, 1].set_title("")
    plt.tight_layout()
    plt.savefig('Figs/SI 8 model lake-level parameters samples.png', dpi=300)


    var_name1s = ['alpha_0', 'beta1_0', 'beta2_0', 'sigma_alpha', 'sigma_beta1', 'sigma_beta2', 'sigma_sigma']
    var_name2s = ['alpha', 'beta1', 'beta2', 'sigma']
    rhat_values = summary['r_hat']
    print(rhat_values.loc[var_name1s])
    print(rhat_values.loc[['{}[{}]'.format(var_name2s[0], i) for i in range(577)]])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), width_ratios=[5, 3], sharey='row')
    plt.sca(axes[0])
    plt.bar(range(len(var_name1s)), rhat_values.loc[var_name1s], color=fill_c, width=0.75, edgecolor=line_c, linewidth=1, alpha=0.7)
    plt.ylim(0.9, 1.1)
    plt.axhline(1.0, color='grey', ls='--', lw=0.5)
    plt.axhline(1.05, color=mark_c, linestyle='--', lw=1)
    plt.xticks(range(len(var_name1s)), [label_dic[v] for v in var_name1s])
    plt.ylabel(r'$\hat{R}$')
    plt.title('Hyperparameters', fontsize=10, fontweight='bold')
    bar_mean_values = [np.mean(rhat_values.loc[['{}[{}]'.format(v, i) for i in range(577)]].values) for v in var_name2s]
    bar_std_values = [np.std(rhat_values.loc[['{}[{}]'.format(v, i) for i in range(577)]].values) for v in var_name2s]
    plt.sca(axes[1])
    plt.bar(range(len(var_name2s)), bar_mean_values, color=fill_c, yerr=bar_std_values, width=0.7, edgecolor=line_c, linewidth=1, alpha=0.7)
    plt.ylim(0.9, 1.1)
    plt.axhline(1.0, color='grey', ls='--', lw=0.5)
    plt.axhline(1.05, color=mark_c, linestyle='--', lw=1)
    plt.xticks(range(len(var_name2s)), [label_dic[v] for v in var_name2s])
    plt.title('Lake-level parameters', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Figs/SI 9 model Rhat Distribution.png', dpi=300)


def fig9_1_lake_paras_corr():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    i_study = 9

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values


    p_dic = {
        'alpha': alpha_post,
        'beta1': beta1_post,
        'beta2': beta2_post,
        'sigma': sigma_post,
    }
    label_dic = [
        r"$\beta_{0,i}$",
        r"$\beta_{1,i}$",
        r"$\beta_{2,i}$",
        r"$\tau_{i}$",
    ]

    var_ls = ['alpha', 'beta1', 'beta2', 'sigma']
    pair_ls = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    r_df = pd.DataFrame(index=range(n_lakes))
    for i_l in range(n_lakes):
        alpha, beta1, beta2, sigma = p_dic['alpha'][:, i_l], p_dic['beta1'][:, i_l], p_dic['beta2'][:, i_l], p_dic['sigma'][:, i_l]
        ls = [alpha, beta1, beta2, sigma]
        print(i_l, len(alpha))
        for pair in pair_ls:
            x, y = ls[pair[0]], ls[pair[1]]
            r = np.corrcoef(x, y)[0, 1]
            r_df.loc[i_l, '{}-{}'.format(pair[0], pair[1])] = r


    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    axes = axes.flatten()
    for i, pair in enumerate(pair_ls):
        plt.sca(axes[i])
        x = r_df.loc[:, '{}-{}'.format(pair[0], pair[1])].values
        plt.hist(r_df.loc[:, '{}-{}'.format(pair[0], pair[1])].values, bins=20, density=True, color=fill_c, alpha=0.7, edgecolor=line_c, linewidth=0.5)
        plt.axvline(0, color=line_c, linestyle='--', linewidth=1, alpha=0.9)
        if i in [0, 3]:
            plt.ylabel('Density')
        plt.xlim(-1, 1)
        plt.xlabel('Intra-lake correlations ($r$)\n({} vs. {})'.format(label_dic[pair[0]], label_dic[pair[1]]))

    for ax, label in {axes[0]: 'a', axes[1]: 'b', axes[2]: 'c', axes[3]: 'd', axes[4]: 'e', axes[5]: 'f'}.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    plt.tight_layout()
    plt.savefig('Figs/SI 9_2 Lake-level paras corr.png', dpi=300)


def fig10_model_fitness():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    i_study = 9

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values

    corr_matrix = data_df[['log_TP_centered', 'temp_centered']].corr()
    print(corr_matrix)

    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    # beta3_post = pd.read_csv('results/samples/beta3_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values


    r2_df = pd.DataFrame(index=lake_codes)
    r2_df.index.name = 'Unique Lake'
    pred_ls, mea_ls = [], []
    for lake_id in range(n_lakes):
        print(lake_id, np.percentile(beta1_post[:, lake_id], [5, 50, 95]),
              np.percentile(beta2_post[:, lake_id], [5, 50, 95]), np.percentile(sigma_post[:, lake_id], [5, 50, 95]))
        lake_mask = lake_idx == lake_id
        tp_i = log_TP_centered[lake_mask]
        tp_i_no_center = log_TP[lake_mask]
        temp_i = temp_centered[lake_mask]
        chla_i = chla[lake_mask]
        chla_i = np.log(chla_i)


        lake_data_df = pd.DataFrame(index=range(len(chla_i)))
        lake_data_df['tp'] = tp_i
        lake_data_df['temp'] = temp_i
        lake_data_df['mix'] = tp_i * temp_i
        lake_data_df['chla'] = chla_i

        # n_draw, n_obs
        mu_samples = ((alpha_post[:, lake_id].reshape(-1, 1)
                       + beta1_post[:, lake_id].reshape(-1, 1) * tp_i.reshape(1, -1))
                      + beta2_post[:, lake_id].reshape(-1, 1) * temp_i.reshape(1, -1))
        r2 = r2_score(chla_i, np.mean(mu_samples, axis=0))
        r2_df.loc[lake_codes[lake_id], 'r2'] = r2
        r2_df.loc[lake_codes[lake_id], 'alpha'] = np.mean(alpha_post[:, lake_id])
        r2_df.loc[lake_codes[lake_id], 'beta1'] = np.mean(beta1_post[:, lake_id])
        r2_df.loc[lake_codes[lake_id], 'beta2'] = np.mean(beta2_post[:, lake_id])
        r2_df.loc[lake_codes[lake_id], 'sigma'] = np.mean(sigma_post[:, lake_id])
        mea_ls.extend(np.exp(chla_i))
        pred_ls.extend(np.exp(np.mean(mu_samples, axis=0)))


    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axes[0])
    overall_r2 = r2_score(np.log(mea_ls), np.log(pred_ls))
    plt.scatter(mea_ls, pred_ls, alpha=0.3, c=scat_c, edgecolors=line_c, s=20, linewidths=.25, zorder=10)
    plt.text(0.05, 0.8, 'Overall R$^2$: {:.2f}'.format(overall_r2), transform=axes[0].transAxes, fontsize=10, va='top', ha='left')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Measured Chla (µg/L)')
    plt.ylabel('Predicted mean Chla (µg/L)')
    y_max = max(max(mea_ls), max(pred_ls))
    plt.plot([1, y_max], [1, y_max], ls='--', c=line_c, lw=2, alpha=0.5, zorder=100)

    print(np.median(r2_df['r2'].values))
    plt.sca(axes[1])
    plt.hist(r2_df['r2'].values, bins=20, density=True, color=fill_c, alpha=0.7, edgecolor=line_c, linewidth=0.5)
    plt.xlabel('Intra-lake R$^2$')
    plt.ylabel('Density')

    for ax, label in {axes[0]: 'a', axes[1]: 'b'}.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('Figs/SI 10 model fitness r2.png', dpi=300)


def fig11_paras_dist_class():
    data_df, meta_df = get_lake_data(easy_get=True)

    i_study = 9
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])


    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta3_post = pd.read_csv('results/samples/beta3_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values


    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)
    # beta3_post_low, beta3_post_median, beta3_post_high = np.percentile(beta3_post, [5, 50, 95], axis=0)
    n_beta1_sig_pos, n_beta1_pos, n_beta1_sig_neg = (beta1_post_low > 0).sum(), (beta1_post_median > 0).sum(), (
                beta1_post_high < 0).sum()
    n_beta2_sig_pos, n_beta2_pos, n_beta2_sig_neg = (beta2_post_low > 0).sum(), (beta2_post_median > 0).sum(), (
                beta2_post_high < 0).sum()
    # n_beta3_sig_pos, n_beta3_pos, n_beta3_sig_neg = (beta3_post_low > 0).sum(), (beta3_post_median > 0).sum(), (beta3_post_high < 0).sum()
    print('beta1 {} pos, {} sig pos, {} sig neg'.format(n_beta1_pos / n_lakes, n_beta1_sig_pos / n_lakes,
                                                        n_beta1_sig_neg / n_lakes))
    print('beta2 {} pos, {} sig pos, {} sig neg'.format(n_beta2_pos / n_lakes, n_beta2_sig_pos / n_lakes,
                                                        n_beta2_sig_neg / n_lakes))
    # print('beta3 {} pos, {} sig pos, {} sig neg'.format(n_beta3_pos / n_lakes, n_beta3_sig_pos / n_lakes, n_beta3_sig_neg / n_lakes))


    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    # post_df['beta3'] = beta3_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta1_sig_neg'] = beta1_post_high < 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0
    post_df['beta1_no_sig'] = np.where((post_df['beta1_sig_pos'] == 0) & (post_df['beta1_sig_neg'] == 0), 1, 0)
    post_df['beta2_no_sig'] = np.where((post_df['beta2_sig_pos'] == 0) & (post_df['beta2_sig_neg'] == 0), 1, 0)

    post_df['beta1_trend'] = post_df['beta1_sig_pos'].astype(int) - post_df['beta1_sig_neg'].astype(int)
    post_df['beta2_trend'] = post_df['beta2_sig_pos'].astype(int) - post_df['beta2_sig_neg'].astype(int)
    counts = post_df.groupby(['beta1_trend', 'beta2_trend']).size().reset_index(name='Count')
    all_combinations = pd.MultiIndex.from_product([[-1, 0, 1], [-1, 0, 1]], names=['beta1_trend', 'beta2_trend'])
    counts = counts.set_index(['beta1_trend', 'beta2_trend']).reindex(all_combinations, fill_value=0).reset_index()

    post_df['G1_PP'] = (post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G2_PN'] = (post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    post_df['G3_NP'] = (~post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G4_NN'] = (~post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    conditions = [post_df['G1_PP'], post_df['G2_PN'], post_df['G3_NP'], post_df['G4_NN']]
    labels = ['PP', 'PN', 'NP', 'NN']
    post_df['response_type'] = np.select(conditions, labels, default='UNCLASSIFIED')
    lake_counts = post_df.groupby('response_type').size()
    print(lake_counts)

    def classify(row):
        if row['beta1_sig_pos'] and row['beta2_sig_pos']:
            return 'PP'
        elif row['beta1_sig_pos'] and not row['beta2_sig_pos']:
            return 'PN'
        elif not row['beta1_sig_pos'] and row['beta2_sig_pos']:
            return 'NP'
        else:
            return 'NN'

    post_df['Class'] = post_df.apply(classify, axis=1)
    class_map = {'NN': 0, 'NP': 1, 'PN': 2, 'PP': 3}
    post_df['Class_num'] = post_df['Class'].map(class_map).astype(int)


    fig = plt.figure(figsize=(8, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 6], width_ratios=[4, 4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])


    ax_cols = ['beta1', 'beta2']
    ax_titles = [r'TP coefficient ($\beta_1$)', r'T coefficient ($\beta_2$)']
    ax_inset_titles = [r"Sig. of $\beta_{1,i}$", r"Sig. of $\beta_{2,i}$"]
    ax_pie_colors = [['#D98880', '#85929E', '#ECF0F1'], ['#A2D9CE', '#85929E', '#ECF0F1']]
    ax_x_labels = [r'Posterior distribution of hyperparameter $\beta_1$',
                   r'Posterior distribution of hyperparameter $\beta_2$']
    for i_a, ax in enumerate([ax1, ax2]):
        plt.sca(ax)
        data = post_df[ax_cols[i_a]].values

        plt.hist(data, bins=25, density=True, alpha=0.6, color='#BDC3C7', edgecolor='white')

        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        plt.plot(x_range, kde(x_range), color='#1C2833', lw=2, alpha=1)

        plt.axvline(0, color='#7F8C8D', linestyle='--', lw=0.5, alpha=1)


        ax_inset = ax.inset_axes([0.6, 0.5, 0.4, 0.4])
        labels = ['+', '—', 'n.s.']
        g_counts = [len(post_df[post_df['{}_sig_pos'.format(ax_cols[i_a])] > 0]),
                  len(post_df[post_df['{}_sig_neg'.format(ax_cols[i_a])] > 0])]
        g_counts.append(n_lakes - g_counts[0] - g_counts[1])
        ax_inset.pie(g_counts, labels=labels, colors=ax_pie_colors[i_a], autopct='%1.0f%%', textprops={'fontsize': 7},
                     startangle=90)
        ax_inset.set_title(ax_inset_titles[i_a], fontsize=8)

        plt.title(ax_titles[i_a])
        plt.xlabel(ax_x_labels[i_a])
        if i_a == 0:
            plt.ylabel("Density")


    plt.sca(ax3)
    tick_labels = ['—', 'n.s.', '+']
    tick_values = [-1, 0, 1]
    bubble_scale = 30
    print(counts)
    c9_ls = ['#85929e', '#ecf0f1', '#a2d9ce',
           '#ecf0f1','#ecf0f1', '#a2d9ce',
            '#d98880', '#d98880', '#b03a2e']
    scatter = plt.scatter(
        counts['beta1_trend'],
        counts['beta2_trend'],
        s=counts['Count'] * bubble_scale,
        c=c9_ls,
        alpha=1,
        edgecolors='none',
        linewidths=0.6,
        zorder=2
    )
    for i, row in counts.iterrows():
        if row['Count'] > -1:
            plt.text(
                row['beta1_trend'],
                row['beta2_trend'],
                f"n={int(row['Count'])}",
                ha='center',
                va='center',
                fontsize=9,
                zorder=100
            )

    plt.plot([0.55, 1.45, 1.45, 0.55, 0.55], [0.55, 0.55, 1.45, 1.45, 0.55], c=c9_ls[8], lw=2, ls='-')
    plt.text(1.4, 1.4, 'PP', fontsize=12, va='top', ha='right')
    plt.plot([0.55, 1.45, 1.45, 0.55, 0.55], [-1.45, -1.45, 0.45, 0.45, -1.45], c=c9_ls[7], lw=2, ls='-')
    plt.text(1.4, 0.4, 'PN', fontsize=12, va='top', ha='right')
    plt.plot([-1.25, 0.45, 0.45, -1.25, -1.25], [0.55, 0.55, 1.45, 1.45, 0.55], c=c9_ls[2], lw=2, ls='-')
    plt.text(0.4, 1.4, 'NP', fontsize=12, va='top', ha='right')
    plt.plot([-1.25, 0.45, 0.45, -1.25, -1.25], [-1.45, -1.45, 0.45, 0.45, -1.45], c=c9_ls[1], lw=2, ls='-')
    plt.text(0.4, 0.4, 'NN', fontsize=12, va='top', ha='right')

    plt.xticks(tick_values, tick_labels, fontsize=8)
    plt.yticks(tick_values, tick_labels, fontsize=8)
    plt.xlim(-1.6, 1.6)
    plt.ylim(-1.6, 1.6)
    plt.xlabel(ax_inset_titles[0], fontsize=9)
    plt.ylabel(ax_inset_titles[1], fontsize=9)


    for ax, label in {ax1: 'a', ax2: 'b', ax3: 'c'}.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('figs/SI 11 paras class.png', dpi=300)


def fig12_class_diff_box():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]
    i_study = 9

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values

    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values


    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0

    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Latitude'] = np.abs(meta_df['Latitude'])

    print(meta_df.columns)
    cols = ['Latitude', 'Altitude (m)',
            'temp_mean', 'sr_mean', 'pr_mean', 'ws_mean',
            'Depth_avg', 'Lake_area', 'Res_time', 'Shore_dev', 'Wind_exposure_idx',
            'TP_mean', 'Ch_mean'
            ]
    abb_dic = {
        'Lat': 'Latitude',
        'log(Alt)': 'Altitude (m)',

        'log(T)': 'temp_mean',
        'log(SR)': 'sr_mean',
        'log(WS)': 'ws_mean',
        'log(Pr)': 'pr_mean',

        'log(Depth)': 'Depth_avg',
        'log(Area)': 'Lake_area',
        'log(RT)': 'Res_time',
        'log(SDI)': 'Shore_dev',
        'log(WEI)': 'Wind_exposure_idx',

        'log(TP)': 'TP_mean',
        'log(Chla)': 'Ch_mean',
    }
    abb1_dic = {v: k for k, v in abb_dic.items()}

    meta_df = meta_df.dropna(subset=cols)
    for col in cols[1:]:
        meta_df[col] = np.clip(meta_df[col], 1, None)
        meta_df[col] = np.log10(meta_df[col])
    meta_df = meta_df.dropna(subset=cols)


    post_df = post_df.join(meta_df, on='Unique Lake')
    print(post_df.head(), post_df.columns)


    post_df['G1_PP'] = (post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G2_PN'] = (post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    post_df['G3_NP'] = (~post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G4_NN'] = (~post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    conditions = [post_df['G4_NN'], post_df['G3_NP'], post_df['G2_PN'], post_df['G1_PP']]
    labels = ['NN', 'NP', 'PN', 'PP']
    post_df['response_type'] = np.select(conditions, labels, default='UNCLASSIFIED')

    compare_cols = cols[:-1]
    group_order = labels
    fig, axes = plt.subplots(4, 3, figsize=(8, 8), sharex='col')
    axes = axes.flatten()
    colors = ['#f2f4f4', '#a2d9ce', '#d98880', '#b03a2e']
    for i, var in enumerate(compare_cols):
        ax = axes[i]
        plt.sca(axes[i])

        for j, g in enumerate(group_order):
            box_values = post_df.loc[post_df['response_type'] == g, var].dropna().values
            bp =plt.boxplot(box_values, positions=[j], widths=0.7, showfliers=True, patch_artist=True,
                        medianprops={'color': line_c, 'linewidth': 1.},
                        whiskerprops={'linestyle': '--', 'color': line_c},
                        flierprops={'marker': 'o', 'markerfacecolor': colors[j], 'markersize': 3, 'alpha': 0.3, 'markeredgecolor': 'none'})
            for patch in bp['boxes']:
                patch.set_facecolor(colors[j])
                patch.set_alpha(0.7)
                patch.set_edgecolor(line_c)
                patch.set_linewidth(1)

        plt.ylabel(abb1_dic[var])
        plt.xticks(range(len(group_order)), group_order)

        groups = [post_df.loc[post_df['response_type'] == g, var].dropna().values for g in group_order]
        H_stat, p_kw = kruskal(*groups)
        kw_text = r"$p_{K-W} < 0.001$" if p_kw < 0.001 else r"$p_{K-W}$" + ' = {:.3f}'.format(p_kw)
        ax.text(0.02, 0.98, kw_text, transform=ax.transAxes, va='top', fontsize=9)

        y_max_all = max([g.max() if len(g) > 0 else 0 for g in groups])
        if p_kw < 0.05:
            pairs = list(combinations(range(len(group_order)), 2))
            pvals = []
            for (i1, i2) in pairs:
                vals1 = groups[i1]
                vals2 = groups[i2]
                stat, p = mannwhitneyu(vals1, vals2, alternative='two-sided')
                pvals.append(p)
            reject, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')
            y_max = post_df[var].max()
            y_min, y_range = post_df[var].min(), post_df[var].max() - post_df[var].min()
            current_y = y_max + y_range * 0.02
            step = y_range * 0.1

            for k, (i1, i2) in enumerate(pairs):
                if reject[k]:
                    stars = p_to_star(pvals_corrected[k])
                    if stars == 'ns': continue
                    ax.plot([i1, i2], [current_y, current_y], color='black', lw=0.8)
                    cap_h = step * 0.2
                    ax.plot([i1, i1], [current_y - cap_h, current_y], color='black', lw=0.8)
                    ax.plot([i2, i2], [current_y - cap_h, current_y], color='black', lw=0.8)
                    ax.text((i1 + i2) / 2, current_y + y_range * 0.01, stars, ha='center', va='center', fontsize=9)
                    current_y += step
            ax.set_ylim(y_min - y_range * 0.05, current_y + step * 2)
    plt.tight_layout()
    plt.savefig('figs/SI 12 paras class.png', dpi=300)


def fig13_tree_class():
    data_df, meta_df = get_lake_data(easy_get=True)

    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]

    i_study = 9

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values

    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0

    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Latitude'] = np.abs(meta_df['Latitude'])

    print(meta_df.columns)
    cols = ['Latitude', 'Altitude (m)',
            'temp_mean', 'sr_mean', 'pr_mean', 'ws_mean',
            'Depth_avg', 'Lake_area', 'Res_time', 'Shore_dev', 'Wind_exposure_idx',
            'TP_mean', 'Ch_mean'
            ]
    abb_dic = {
        'Lat': 'Latitude',
        'log(Alt)': 'Altitude (m)',

        'log(T)': 'temp_mean',
        'log(SR)': 'sr_mean',
        'log(WS)': 'ws_mean',
        'log(Pr)': 'pr_mean',

        'log(Depth)': 'Depth_avg',
        'log(Area)': 'Lake_area',
        'log(RT)': 'Res_time',
        'log(SDI)': 'Shore_dev',
        'log(WEI)': 'Wind_exposure_idx',

        'log(TP)': 'TP_mean',
        'log(Chla)': 'Ch_mean',
    }
    abb1_dic = {v: k for k, v in abb_dic.items()}

    meta_df = meta_df.dropna(subset=cols)
    print('x lakes', len(meta_df))
    for col in cols[1:]:
        meta_df[col] = np.clip(meta_df[col], 1, None)
        meta_df[col] = np.log10(meta_df[col])
    meta_df = meta_df.dropna(subset=cols)
    print('add x lakes', len(meta_df))

    post_df = post_df.join(meta_df, on='Unique Lake')
    print(post_df.head(), post_df.columns)


    post_df['G1_PP'] = (post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G2_PN'] = (post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    post_df['G3_NP'] = (~post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G4_NN'] = (~post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    conditions = [post_df['G1_PP'], post_df['G2_PN'], post_df['G3_NP'], post_df['G4_NN']]
    labels = ['PP', 'PN', 'NP', 'NN']
    post_df['response_type'] = np.select(conditions, labels, default='UNCLASSIFIED')
    lake_counts = post_df.groupby('response_type').size()
    print(lake_counts)

    def classify(row):
        if row['beta1_sig_pos'] and row['beta2_sig_pos']:
            return 'PP'
        elif row['beta1_sig_pos'] and not row['beta2_sig_pos']:
            return 'PN'
        elif not row['beta1_sig_pos'] and row['beta2_sig_pos']:
            return 'NP'
        else:
            return 'NN'

    post_df['Class'] = post_df.apply(classify, axis=1)
    class_map = {'NN': 0, 'NP': 1, 'PN': 2, 'PP': 3}
    post_df['Class_num'] = post_df['Class'].map(class_map).astype(int)

    x_cols = [
        'Lake_area',
        'Depth_avg',
        'temp_mean',
        'Wind_exposure_idx',
        'Res_time',
        'TP_mean',
    ]
    X = post_df[x_cols]
    y = post_df['Class_num']
    n_depth = 5
    clf = DecisionTreeClassifier(max_depth=n_depth, min_samples_leaf=10, random_state=911)
    clf.fit(X, y)


    def get_node_data(clf, feature_names):
        from sklearn.tree import _tree
        tree_ = clf.tree_
        node_data = {}
        print(tree_.feature)

        def recurse(node, depth, x, width):
            counts = tree_.value[node][0]
            node_data[node] = {
                'x': x,
                'y': 1 - depth * (1 / n_depth),
                'counts': counts,
                'samples': tree_.n_node_samples[node],
                'feature': feature_names[tree_.feature[node]] if tree_.feature[node] != _tree.TREE_UNDEFINED else None,
                'threshold': tree_.threshold[node]
            }

            left = tree_.children_left[node]
            right = tree_.children_right[node]

            if left != _tree.TREE_LEAF:
                recurse(left, depth + 1, x - width / 2, width / 2)
                node_data[node]['left_child'] = left
            if right != _tree.TREE_LEAF:
                recurse(right, depth + 1, x + width / 2, width / 2)
                node_data[node]['right_child'] = right

        recurse(0, 0, 0.5, 0.5)
        return node_data


    plt.figure(figsize=[18, 8])
    ax = plt.gca()
    colors = ['#F2F4F4', '#A2D9CE', '#D98880', '#B03A2E']
    name_dic = {'TP_mean': 'TP', 'temp_mean': 'T', 'Lake_area': 'Area', 'Depth_avg': 'Depth', 'Wind_exposure_idx': "WEI", 'Res_time': 'RT',
                'Latitude': 'Lat', 'Altitude (m)': 'Alt', 'pr_mean': 'Pr', 'ws_mean': 'WS', 'sr_mean': 'SR', 'Shore_dev': 'SDI'}
    unit_dic = {'TP_mean': ' µg/L', 'temp_mean': '°C', 'Lake_area': ' km$^2$', 'Depth_avg': ' m', 'Wind_exposure_idx': "×10$^3$ m$^2$/s", 'Res_time': ' d',
                'Latitude': '°', 'Altitude (m)': ' m', 'pr_mean': ' mm/d', 'ws_mean': ' m/s', 'sr_mean': 'W/m$^2$', 'Shore_dev': ''}
    node_data = get_node_data(clf, x_cols)
    all_samples = [info['samples'] for info in node_data.values()]
    min_s, max_s = min(all_samples), max(all_samples)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    for node, info in node_data.items():
        if 'left_child' in info:
            child = node_data[info['left_child']]
            y_median = (info['y'] + child['y']) / 2
            ax.plot([info['x'], info['x']], [y_median + 0.01, y_median], color='k', lw=0.5, zorder=1)
            ax.plot([info['x'], child['x']], [y_median, y_median], color='k', lw=0.5, zorder=1)
            ax.plot([child['x'], child['x']], [y_median, y_median - 0.01], color='k', lw=0.5, zorder=1)
            if info['y'] == 1:
                ax.text((info['x'] + child['x']) / 2, y_median + 0.01, "True", ha='center', fontsize=7, color='k')
        if 'right_child' in info:
            child = node_data[info['right_child']]
            y_median = (info['y'] + child['y']) / 2
            ax.plot([info['x'], info['x']], [y_median + 0.01, y_median], color='k', lw=0.5, zorder=1)
            ax.plot([info['x'], child['x']], [y_median, y_median], color='k', lw=0.5, zorder=1)
            ax.plot([child['x'], child['x']], [y_median, y_median - 0.01], color='k', lw=0.5, zorder=1)
            if info['y'] == 1:
                ax.text((info['x'] + child['x']) / 2, y_median + 0.01, "False", ha='center', fontsize=7, color='k')

    for node, info in node_data.items():
        MIN_PIE_SIZE = 0.04
        MAX_PIE_SIZE = 0.16
        size = np.interp(info['samples'], [min_s, max_s], [MIN_PIE_SIZE, MAX_PIE_SIZE])

        ax_inset = ax.inset_axes([info['x'] - size / 2, info['y'] - size / 2, size, size], transform=ax.transData)
        ax_inset.pie(info['counts'], colors=colors, startangle=90)

        if info['feature']:
            feature_name, feature_value = name_dic[info['feature']], np.power(10, info['threshold'])
            ax.text(info['x'], info['y'] - (1 / n_depth) * 0.4,
                    '{} < {:.1f}{}'.format(feature_name, feature_value, unit_dic[info['feature']]), ha='center',
                    fontsize=8, color='k')

        ax.text(info['x'], info['y'] + (1 / n_depth) * 0.36, f"n={int(info['samples'])}",
                ha='center', fontsize=8, color='k')

    ax_leg = ax.inset_axes([0.87, 0.88, 0.06, 0.12])
    matrix_colors = [
        [colors[0], colors[2]],
        [colors[1], colors[3]]
    ]
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j, i), 1, 1, facecolor=matrix_colors[i][j], edgecolor='white', lw=1)
            ax_leg.add_patch(rect)
    ax_leg.set_xlim(0, 2)
    ax_leg.set_ylim(0, 2)
    ax_leg.set_xticks([0.5, 1.5])
    ax_leg.set_yticks([0.5, 1.5])
    ax_leg.set_xticklabels(['n.s. / —', '+'], fontsize=6.5)
    ax_leg.set_yticklabels(['n.s. / —', '+'], fontsize=6.5)
    ax_leg.set_xlabel(r'$\beta_{1,i}$', fontsize=7)
    ax_leg.set_ylabel(r'$\beta_{2,i}$', fontsize=7)
    ax_leg.spines['top'].set_visible(False)
    ax_leg.spines['right'].set_visible(False)
    ax_leg.spines['bottom'].set_visible(False)
    ax_leg.spines['left'].set_visible(False)
    ax_leg.tick_params(size=0)
    ax_leg.grid(False)
    ax.axis('off')


    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.savefig('figs/SI 13 Tree class.png', dpi=300)


def fig1314_class_panel_and_high_nutrient_chl():
    data_df, meta_df = get_lake_data(easy_get=True)

    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]

    i_study = 9

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values

    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0

    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Latitude'] = np.abs(meta_df['Latitude'])

    print(meta_df.columns)
    cols = ['Latitude', 'Altitude (m)',
            'temp_mean', 'sr_mean', 'pr_mean', 'ws_mean',
            'Depth_avg', 'Lake_area', 'Res_time', 'Shore_dev', 'Wind_exposure_idx',
            'TP_mean', 'Ch_mean'
            ]

    meta_df = meta_df.dropna(subset=cols)
    print('x lakes', len(meta_df))
    meta_df = meta_df.dropna(subset=cols)
    print('add x lakes', len(meta_df))


    post_df = post_df.join(meta_df, on='Unique Lake')


    for lake_id in range(n_lakes):
        lake_mask = lake_idx == lake_id
        tp_i = log_TP_centered[lake_mask]
        tp_i_no_center = log_TP[lake_mask]
        temp_i = temp_centered[lake_mask]
        chla_i = chla[lake_mask]
        chla_i = np.log(chla_i)
        tp_i_nolog = np.exp(tp_i_no_center)
        temp_i_nocenter = temp_i + 10

        slope, intercept, r_value, p_value, std_err = linregress(temp_i, tp_i)
        post_df.loc[lake_codes[lake_id], 'x12_r'] = r_value
        post_df.loc[lake_codes[lake_id], 'temp_cv_used'] = np.std(temp_i_nocenter) / np.mean(temp_i_nocenter)
        post_df.loc[lake_codes[lake_id], 'TP_cv_used'] = np.std(tp_i_nolog) / np.mean(tp_i_nolog)
        post_df.loc[lake_codes[lake_id], 'TP_mean'] = np.exp(np.mean(tp_i_no_center))
        post_df.loc[lake_codes[lake_id], 'temp_mean'] = np.mean(temp_i + 10)
    print(post_df.head(), post_df.columns)


    post_df['G1_PP'] = (post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G2_PN'] = (post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    post_df['G3_NP'] = (~post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G4_NN'] = (~post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    conditions = [post_df['G1_PP'], post_df['G2_PN'], post_df['G3_NP'], post_df['G4_NN']]
    labels = ['PP', 'PN', 'NP', 'NN']
    post_df['response_type'] = np.select(conditions, labels, default='UNCLASSIFIED')
    lake_counts = post_df.groupby('response_type').size()
    print(lake_counts)

    def classify(row):
        if row['beta1_sig_pos'] and row['beta2_sig_pos']:
            return 'PP'
        elif row['beta1_sig_pos'] and not row['beta2_sig_pos']:
            return 'PN'
        elif not row['beta1_sig_pos'] and row['beta2_sig_pos']:
            return 'NP'
        else:
            return 'NN'

    post_df['Class'] = post_df.apply(classify, axis=1)
    class_map = {'NN': 0, 'NP': 1, 'PN': 2, 'PP': 3}
    post_df['Class_num'] = post_df['Class'].map(class_map).astype(int)


    plt.figure(figsize=[8, 8])
    ax = plt.gca()
    for class_name in class_names:
        sub_df = post_df[post_df['Class'] == class_name]
        x, y = sub_df['TP_mean'].values, sub_df['temp_mean'].values
        plt.scatter(x, y, s=25, alpha=1, color=class_c_dic[class_name], linewidth=0.5, edgecolors=line_c, label=class_name)

    x_min, x_max = 0.1, 2.3
    y_min, y_max = 0, 27
    split_c, alpha, lw = scat_c, 0.6, 1
    plt.plot([10**1.0476744771003723, 10**1.0476744771003723], [y_min, y_max], color=split_c, linestyle='-', lw=lw, alpha=alpha)
    plt.plot([10**0.1, 10**1.0476744771003723], [7.336688280105591, 7.336688280105591], color=split_c, linestyle='-', lw=lw, alpha=alpha)
    plt.plot([10**1.4969202876091003, 10**1.4969202876091003], [0, 16.1571364402771], color=split_c, linestyle='-', lw=lw, alpha=alpha)
    plt.plot([10**1.0476744771003723, 10**x_max], [16.1571364402771, 16.1571364402771], color=split_c, linestyle='-', lw=lw, alpha=alpha)
    plt.plot([10**1.9525858759880066, 10**1.9525858759880066], [16.1571364402771, y_max], color=split_c, linestyle='-', lw=lw, alpha=alpha)

    plt.text(3.2, 0.5, 'A1 (n=61, 56% NN)', va='bottom', ha='left', fontsize=9, fontweight='bold')
    plt.text(3.2, 9, 'A2 (n=40, 55% PN)', va='bottom', ha='left', fontsize=9, fontweight='bold')
    plt.text(11.4, 0.5, 'A3 (n=266, 64% PN)', va='bottom', ha='left', fontsize=9, fontweight='bold')
    plt.text(31.7, 0.5, 'A4 (n=176, 44% PN, 44% PP)', va='bottom', ha='left', fontsize=9, fontweight='bold')
    plt.text(11.4, 16.7, 'A5 (n=16, 56% PN)', va='bottom', ha='left', fontsize=9, fontweight='bold')
    plt.text(90, 16.7, 'A6 (n=18, 83% NN)', va='bottom', ha='left', fontsize=9, fontweight='bold')
    plt.xlim(3, None)
    plt.ylim(y_min, None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xscale('log')
    plt.xlabel('Lake mean TP (µg/L)')
    plt.ylabel('Lake mean T (°C)')
    plt.legend(loc='upper left')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('figs/SI 1314-1 Classes in TP-T space 2.png', dpi=300)


    tp_thre = 11.2
    t_thres = [7.336688280105591, 16.2]
    title_dic = {}
    for i_fig in range(2):
        sub_df = post_df[post_df['TP_mean'] > tp_thre] if i_fig else post_df[post_df['TP_mean'] < tp_thre]
        if i_fig:
            left1_post_df = sub_df[(sub_df['temp_mean'] < t_thres[i_fig]) & (sub_df['TP_mean'] < 31.4)]
            left2_post_df = sub_df[(sub_df['temp_mean'] < t_thres[i_fig]) & (sub_df['TP_mean'] > 31.4)]
            righ1_post_df = sub_df[(sub_df['temp_mean'] > t_thres[i_fig]) & (sub_df['TP_mean'] < 89.7)]
            righ2_post_df = sub_df[(sub_df['temp_mean'] > t_thres[i_fig]) & (sub_df['TP_mean'] > 89.7)]
            title_dic['Lakes with mean TP > 11 µg/L\nand mean T < {:.0f}°C\n and mean TP < 31 µg/L'.format(t_thres[i_fig])] = left1_post_df
            title_dic['Lakes with mean TP > 11 µg/L\nand mean T < {:.0f}°C\n and mean TP > 31 µg/L'.format(t_thres[i_fig])] = left2_post_df
            title_dic['Lakes with mean TP > 11 µg/L\nand mean T > {:.0f}°C\n and mean TP < 90 µg/L'.format(t_thres[i_fig])] = righ1_post_df
            title_dic['Lakes with mean TP > 11 µg/L\nand mean T > {:.0f}°C\n and mean TP > 90 µg/L'.format(t_thres[i_fig])] = righ2_post_df
        else:
            left_post_df = sub_df[sub_df['temp_mean'] < t_thres[i_fig]]
            righ_post_df = sub_df[sub_df['temp_mean'] > t_thres[i_fig]]
            title_dic['Lakes with mean TP < 11 µg/L\nand mean T < {:.0f}°C'.format(t_thres[i_fig])] = left_post_df
            title_dic['Lakes with mean TP < 11 µg/L\nand mean T > {:.0f}°C'.format(t_thres[i_fig])] = righ_post_df
    fig, axes = plt.subplots(3, 2, sharey='row', figsize=(8, 9))
    axes = axes.flatten()
    abc_ls = ['a', 'b', 'c', 'd', 'e', 'f']
    for i_c, (title, df) in enumerate(title_dic.items()):
        ax = axes[i_c]
        plt.sca(ax)

        plt.axhline(0, color=line_c, ls='--', lw=0.5, alpha=.8)

        groups = []
        x_ticklabels = []
        for i_x, class_name in enumerate(class_names):
            sub_df = df[df['Class'] == class_name]
            y = sub_df['x12_r'].values
            bp = plt.boxplot(y, positions=[i_x], widths=0.7, showfliers=True, patch_artist=True,
                             medianprops={'color': line_c, 'linewidth': 1.},
                             whiskerprops={'linestyle': '--', 'color': line_c},
                             flierprops={'marker': 'o', 'markerfacecolor': class_c_dic[class_name], 'markersize': 3, 'alpha': 0.3, 'markeredgecolor': 'none'})
            for patch in bp['boxes']:
                patch.set_facecolor(class_c_dic[class_name])
                patch.set_alpha(0.7)
                patch.set_edgecolor(line_c)
                patch.set_linewidth(1)
            if len(y) > 1:
                groups.append(y)
            x_ticklabels.append('{}\n({:.0f}%)'.format(class_name, (len(sub_df) / len(df)) * 100))

        H_stat, p_kw = kruskal(*groups)
        kw_text = r"$p_{K-W} < 0.001$" if p_kw < 0.001 else r"$p_{K-W}$" + ' = {:.3f}'.format(p_kw)
        ax.text(0.02, 0.98, kw_text, transform=ax.transAxes, va='top', fontsize=9)

        plt.xticks(range(len(class_names)), x_ticklabels)
        plt.title('A{} (n={})'.format(i_c+1, len(df)), fontweight='bold', fontsize=9)
        if i_c in [0, 2, 4]:
            plt.ylabel('Intra-lake correlations ($r$)\n(TP vs. T)')
        plt.text(0.95, 0.95, abc_ls[i_c], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    plt.savefig('figs/SI 1314-5 x12_r in response types under 6 areas.png', dpi=300)


    plt.figure(figsize=(8, 7))
    ax = plt.gca()
    t = 0.3
    r_ranges = [[-1, -t], [-t, t], [t, 1]]
    x_ticks = ['$r$ < -{}'.format(t), '|$r$| < {}'.format(t), '$r$ > {}'.format(t)]
    y_ticks = []
    for i_x, (title, df) in enumerate(title_dic.items()):
        y_ticks.append('A{} (n={})'.format(i_x+1, len(df)))
        for i_y, r_range in enumerate(r_ranges):
            sub_df = df[(df['x12_r'] > r_range[0]) & (df['x12_r'] < r_range[1])]
            counts = []
            for class_name in class_names:
                counts.append(len(sub_df[sub_df['Class'] == class_name]))
            plt.pie(counts, colors=list(class_c_dic.values()), startangle=90, autopct=lambda p: f'{p:.0f}%' if p > 40 else '', textprops={'fontsize': 7}, center=(i_y*2, i_x), radius=0.30)
            plt.text(i_y*2, i_x-0.32, 'n={}'.format(len(sub_df)), va='top', ha='center', fontsize=7)
    plt.xticks([0, 2, 4], x_ticks, rotation=0)
    plt.yticks(range(len(title_dic)), y_ticks, rotation=0)
    plt.xlabel('Intra-lake correlations ($r$)\n(TP vs. T)')
    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 5.5)
    ax.tick_params(which='both', direction='in', length=0)
    plt.tight_layout()
    plt.savefig('figs/SI 1314-6 class pie in x12_r ranges under 6 areas.png', dpi=300)


    x_cols = ['TP_mean', 'temp_mean']
    y_col = 'Ch_mean'
    x_labels = ['Lake mean TP (µg/L)', 'Lake mean T (°C)']
    y_label = 'Lake mean Chla (µg/L)'
    fig, axes = plt.subplots(len(x_cols), 1, figsize=(8, 10 if len(x_cols) == 2 else 8))
    for i_ax, x_col in enumerate(x_cols):
        plt.sca(axes[i_ax])
        x, y = post_df[x_col], post_df[y_col]
        lowess_result = sm.nonparametric.lowess(y, x, frac=.3)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=4, alpha=.2)
        for i_a, (title, df) in enumerate(title_dic.items()):
            x, y = df[x_col], df[y_col]
            plt.scatter(x, y, s=25, alpha=0.75, linewidth=0.5, c=my_colors[i_a], edgecolors=line_c, label='A{} lakes (n={})'.format(i_a+1, len(df)))
            lowess_result = sm.nonparametric.lowess(y, x, frac=.9 if 1 < i_a < 4 else 0.95)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=my_colors[i_a], ls='-', linewidth=2, alpha=1, label='LOWESS for A{}'.format(i_a+1))
        plt.text(0.95, 0.95, abc_ls[i_ax], transform=plt.gca().transAxes, fontsize=12, fontweight='bold', va='top', ha='left')
        plt.xlabel(x_labels[i_ax])
        plt.ylabel(y_label)
        if i_ax == 0:
            plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1, 299)
        plt.legend(loc='upper left', ncol=3)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('figs/SI 1314-2 TP-Chl T-Chl responses in 6 areas.png', dpi=300)


def fig14_relations():
    data_df, meta_df = get_lake_data(easy_get=True)

    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]

    i_study = 9

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values

    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0

    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Latitude'] = np.abs(meta_df['Latitude'])

    print(meta_df.columns)
    cols = ['Latitude', 'Altitude (m)',
            'temp_mean', 'sr_mean', 'pr_mean', 'ws_mean',
            'Depth_avg', 'Lake_area', 'Res_time', 'Shore_dev', 'Wind_exposure_idx',
            'TP_mean', 'Ch_mean'
            ]
    abb_dic = {
        'Lat': 'Latitude',
        'log(Alt)': 'Altitude (m)',

        'log(T)': 'temp_mean',
        'log(SR)': 'sr_mean',
        'log(WS)': 'ws_mean',
        'log(Pr)': 'pr_mean',

        'log(Depth)': 'Depth_avg',
        'log(Area)': 'Lake_area',
        'log(RT)': 'Res_time',
        'log(SDI)': 'Shore_dev',
        'log(WEI)': 'Wind_exposure_idx',

        'log(TP)': 'TP_mean',
        'log(Chla)': 'Ch_mean',
    }
    abb1_dic = {v: k for k, v in abb_dic.items()}

    meta_df = meta_df.dropna(subset=cols)
    print('x lakes', len(meta_df))
    for col in cols[1:]:
        meta_df[col] = np.clip(meta_df[col], 1, None)
        meta_df[col] = np.log10(meta_df[col])
    meta_df = meta_df.dropna(subset=cols)
    print('add x lakes', len(meta_df))

    post_df = post_df.join(meta_df, on='Unique Lake')
    print(post_df.head(), post_df.columns)

    compare_cols = cols[:-1]
    c_names = ['alpha', 'beta1', 'beta2', 'sigma']
    c_name_labels = [r"$\beta_{0}$", r"$\beta_{1}$", r"$\beta_{2}$", r"$\tau$"]
    fig, axes = plt.subplots(len(compare_cols), len(c_names), figsize=(8, 11))
    for i_r, col in enumerate(compare_cols):
        for i_c, c_name in enumerate(c_names):
            ax = axes[i_r, i_c]
            plt.sca(ax)
            x = post_df[col].values
            y = post_df[c_name].values

            ax.set_ylabel(c_name_labels[i_c], fontsize=7, labelpad=1)
            ax.set_xlabel(abb1_dic[col], fontsize=7, labelpad=1)
            ax.scatter(x, y, s=8, alpha=0.3, color=scat_c, linewidth=0.25, edgecolors=line_c)

            lowess_result = sm.nonparametric.lowess(y, x, frac=0.1)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=1., alpha=.7)

            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
            if p_p < 0.05:
                slope, intercept, _, _, _ = stats.linregress(x, y)
                xx = np.linspace(x.min(), x.max(), 100)
                yy = slope * xx + intercept
                if abs(r_p) > 0.3:
                    plt.text(0.02, 1, '{:.2f}'.format(r_p),
                             transform=plt.gca().transAxes, fontsize=9, color=mark_c, va='top', ha='left',
                             fontweight='bold')
                else:
                    plt.text(0.02, 1, '{:.2f}'.format(r_p),
                             transform=plt.gca().transAxes, fontsize=9, color='k', va='top', ha='left', fontweight='bold', alpha=.7)

            ax.tick_params(labelsize=6, width=.5, length=1, pad=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig('figs/SI 14 scatter_corr.png', dpi=300)


def fig15to17_shap_explain():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]

    i_study = 9


    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values

    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0

    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Latitude'] = np.abs(meta_df['Latitude'])

    print(meta_df.columns)
    cols = ['Latitude', 'Altitude (m)',
            'temp_mean', 'sr_mean', 'pr_mean', 'ws_mean',
            'Depth_avg', 'Lake_area', 'Res_time', 'Shore_dev', 'Wind_exposure_idx',
            'TP_mean', 'Ch_mean'
            ]
    abb_dic = {
        'Lat': 'Latitude',
        'log(Alt)': 'Altitude (m)',

        'log(T)': 'temp_mean',
        'log(SR)': 'sr_mean',
        'log(WS)': 'ws_mean',
        'log(Pr)': 'pr_mean',

        'log(Depth)': 'Depth_avg',
        'log(Area)': 'Lake_area',
        'log(RT)': 'Res_time',
        'log(SDI)': 'Shore_dev',
        'log(WEI)': 'Wind_exposure_idx',

        'log(TP)': 'TP_mean',
        'log(Chla)': 'Ch_mean',
    }
    abb1_dic = {v: k for k, v in abb_dic.items()}

    meta_df = meta_df.dropna(subset=cols)
    print('x lakes', len(meta_df))
    for col in cols[1:]:
        meta_df[col] = np.clip(meta_df[col], 1, None)
        meta_df[col] = np.log10(meta_df[col])
    meta_df = meta_df.dropna(subset=cols)
    print('add x lakes', len(meta_df))

    post_df = post_df.join(meta_df, on='Unique Lake')


    for v, k in abb1_dic.items():
        post_df[k] = post_df[v]
    print(post_df.head(), post_df.columns)


    x_cols = [
        'temp_mean',
        'Depth_avg',
        'Res_time',
        'TP_mean',
        'Lake_area',
        'Shore_dev'
            ]
    x_names = ['log(T)', 'log(Depth)', 'log(RT)', 'log(TP)',
               'log(Area)', 'log(SDI)'
               ]
    ys = ['alpha', 'beta1', 'beta2', 'sigma']
    ylim_ls = [[-1.5, 1.5], [-0.5, 0.3], [-0.03, 0.04], [-0.3, 0.4]]
    y_labels = [r"$\beta_{0}$", r"$\beta_{1}$", r"$\beta_{2}$", r"$\tau$"]
    fig_rf_fitness, axes_rf_fitness = plt.subplots(2, 2, figsize=(8, 8))
    axes_rf_fitness = axes_rf_fitness.flatten()
    fig_shap_sum, axes_shap_sum = plt.subplots(2, 2, figsize=(8, 9))
    axes_shap_sum = axes_shap_sum.flatten()
    fig_shap_sca, axes_shap_sca = plt.subplots(len(x_cols), len(ys), figsize=(8, 10))
    text_ls = ['a', 'b', 'c', 'd']
    for i, y in enumerate(ys):
        rf = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(post_df[x_cols].values, post_df[y].values)
        y_pred = rf.predict(post_df[x_cols].values)

        ax = axes_rf_fitness[i]
        plt.sca(ax)
        plt.scatter(post_df[y].values, y_pred, alpha=0.3, c=scat_c, edgecolors=line_c, s=30, linewidths=.5)
        r2 = r2_score(post_df[y].values, y_pred)
        plt.text(0.05, 0.85, 'R$^2$: {:.2f}'.format(r2), transform=ax.transAxes, fontsize=10, color='k', va='top', ha='left')
        y_min, y_max = min(min(post_df[y].values), min(y_pred)), max(max(post_df[y].values), max(y_pred))
        plt.plot([y_min, y_max], [y_min, y_max], ls='--', c=line_c, lw=1., zorder=100)
        ax.tick_params(labelsize=8)
        plt.xlabel(y_labels[i], fontsize=9)
        plt.ylabel('Predicted {}'.format(y_labels[i]), fontsize=9)
        ax.text(0.05, 0.95, text_ls[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')


        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(post_df[x_cols])
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.Series(shap_abs_mean, index=x_cols).sort_values(ascending=False)


        ax = axes_shap_sum[i]
        plt.sca(ax)
        shap.summary_plot(shap_values, post_df[x_cols].values, plot_type="dot", feature_names=x_names, show=False, plot_size=None)
        plt.title(f"Target: {y_labels[i]}", fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.set_xlabel('SHAP value for {}'.format(y_labels[i]), fontsize=9)
        all_axes = plt.gcf().axes
        ax_cbar = all_axes[-1]
        ax_cbar.tick_params(labelsize=8)
        ax_cbar.set_ylabel(ax_cbar.get_ylabel(), fontsize=8, labelpad=2)
        for text in ax_cbar.texts:
            text.set_size(8)

        for i_top, x_col in enumerate(shap_importance.index):
            ax = axes_shap_sca[i_top, i]
            plt.sca(ax)
            x_idx = x_cols.index(x_col)
            plt.scatter(post_df[x_col].values, shap_values[:, x_idx], alpha=0.3, c=scat_c, edgecolors=line_c, s=10, linewidths=.4)
            lowess_result = sm.nonparametric.lowess(shap_values[:, x_idx], post_df[x_col].values, frac=0.25)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=1., alpha=0.7)
            plt.axhline(0, color=line_c, ls='--', lw=0.5, alpha=.8)
            ax.tick_params(labelsize=6, width=.5, length=1, pad=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i_top == 6:
                ax.set_ylabel('SHAP value for {}'.format(y_labels[i]), fontsize=8, labelpad=1)
            if i_top == 0:
                plt.title(f"Target: {y_labels[i]}", fontsize=10, fontweight='bold')
            ax.set_xlabel('Rank {}: {}'.format(i_top + 1, abb1_dic[x_col]), fontsize=7, labelpad=1)
            plt.ylim(ylim_ls[i])

    plt.sca(axes_rf_fitness[0])
    plt.tight_layout()
    plt.savefig('figs/SI 15 RF fitness.png', dpi=300)

    plt.sca(axes_shap_sum[0])
    plt.tight_layout()
    plt.savefig('figs/SI 16 SHAP summary.png', dpi=300)

    plt.sca(axes_shap_sca[0, 0])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45, wspace=0.4)
    plt.savefig('figs/SI 17 SHAP scatter.png', dpi=300)


def fig15to17_shap_explain_for_del_tn():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]
    ini_data_df = copy.deepcopy(data_df)

    i_study = 20


    data_df = data_df.dropna(subset=['Chla (µg/L)', 'TN (µg/L)'])
    data_df['TN (µg/L)_std'] = calc_std_tn(data_df['TP (µg/L)'].values)
    print('tn data', len(data_df))
    data_df = data_df[data_df['TN (µg/L)_std'] < data_df['TN (µg/L)'] * 1]
    print('tn with tp limit data', len(data_df))
    data_df['TP (µg/L)_log'] = np.log10(data_df['TP (µg/L)'])
    data_df['Chla (µg/L)_log'] = np.log10(data_df['Chla (µg/L)'])
    lake_counts = data_df.groupby('Unique Lake').size()
    idx = lake_counts.index.intersection(meta_df.index)
    meta_df.loc[idx, 'n_samples_tp_limit'] = lake_counts.loc[idx].values
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[meta_df['n_samples_tp_limit'] > 15].index)]



    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values

    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0

    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Latitude'] = np.abs(meta_df['Latitude'])

    print(meta_df.columns)
    cols = ['Latitude', 'Altitude (m)',
            'temp_mean', 'sr_mean', 'pr_mean', 'ws_mean',
            'Depth_avg', 'Lake_area', 'Res_time', 'Shore_dev', 'Wind_exposure_idx',
            'TP_mean', 'Ch_mean'
            ]
    abb_dic = {
        'Lat': 'Latitude',
        'log(Alt)': 'Altitude (m)',

        'log(T)': 'temp_mean',
        'log(SR)': 'sr_mean',
        'log(WS)': 'ws_mean',
        'log(Pr)': 'pr_mean',

        'log(Depth)': 'Depth_avg',
        'log(Area)': 'Lake_area',
        'log(RT)': 'Res_time',
        'log(SDI)': 'Shore_dev',
        'log(WEI)': 'Wind_exposure_idx',

        'log(TP)': 'TP_mean',
        'log(Chla)': 'Ch_mean',
    }
    abb1_dic = {v: k for k, v in abb_dic.items()}

    meta_df = meta_df.dropna(subset=cols)
    print('x lakes', len(meta_df))
    for col in cols[1:]:
        meta_df[col] = np.clip(meta_df[col], 1, None)
        meta_df[col] = np.log10(meta_df[col])
    meta_df = meta_df.dropna(subset=cols)

    post_df = post_df.join(meta_df, on='Unique Lake')
    post_df['TP_mean'] = np.log10(post_df['TP_mean'])


    for v, k in abb1_dic.items():
        post_df[k] = post_df[v]
    print(post_df.head(), post_df.columns)


    x_cols = [
        'temp_mean',
        'Depth_avg',
        'Res_time',
        'TP_mean',
        'Lake_area',
        'Shore_dev'
            ]
    x_names = ['log(T)', 'log(Depth)', 'log(RT)', 'log(TP)',
               'log(Area)', 'log(SDI)'
               ]
    ys = ['alpha', 'beta1', 'beta2', 'sigma']
    ylim_ls = [[-1.5, 1.5], [-0.5, 0.3], [-0.03, 0.04], [-0.3, 0.4]]
    y_labels = [r"$\beta_{0}$", r"$\beta_{1}$", r"$\beta_{2}$", r"$\tau$"]
    fig_rf_fitness, axes_rf_fitness = plt.subplots(2, 2, figsize=(8, 8))
    axes_rf_fitness = axes_rf_fitness.flatten()
    fig_shap_sum, axes_shap_sum = plt.subplots(2, 2, figsize=(8, 9))
    axes_shap_sum = axes_shap_sum.flatten()
    fig_shap_sca, axes_shap_sca = plt.subplots(len(x_cols), len(ys), figsize=(8, 10))
    text_ls = ['a', 'b', 'c', 'd']
    for i, y in enumerate(ys):
        rf = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=119,
            n_jobs=-1
        )
        rf.fit(post_df[x_cols].values, post_df[y].values)
        y_pred = rf.predict(post_df[x_cols].values)

        ax = axes_rf_fitness[i]
        plt.sca(ax)
        plt.scatter(post_df[y].values, y_pred, alpha=0.3, c=scat_c, edgecolors=line_c, s=30, linewidths=.5)
        r2 = r2_score(post_df[y].values, y_pred)
        plt.text(0.05, 0.85, 'R$^2$: {:.2f}'.format(r2), transform=ax.transAxes, fontsize=10, color='k', va='top', ha='left')
        y_min, y_max = min(min(post_df[y].values), min(y_pred)), max(max(post_df[y].values), max(y_pred))
        plt.plot([y_min, y_max], [y_min, y_max], ls='--', c=line_c, lw=1., zorder=100)
        ax.tick_params(labelsize=8)
        plt.xlabel(y_labels[i], fontsize=9)
        plt.ylabel('Predicted {}'.format(y_labels[i]), fontsize=9)
        ax.text(0.05, 0.95, text_ls[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(post_df[x_cols])
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.Series(shap_abs_mean, index=x_cols).sort_values(ascending=False)
        print(y, shap_importance)


        ax = axes_shap_sum[i]
        plt.sca(ax)
        shap.summary_plot(shap_values, post_df[x_cols].values, plot_type="dot", feature_names=x_names, show=False, plot_size=None)
        plt.title(f"Target: {y_labels[i]}", fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.set_xlabel('SHAP value for {}'.format(y_labels[i]), fontsize=9)
        all_axes = plt.gcf().axes
        ax_cbar = all_axes[-1]
        ax_cbar.tick_params(labelsize=8)
        ax_cbar.set_ylabel(ax_cbar.get_ylabel(), fontsize=8, labelpad=2)
        for text in ax_cbar.texts:
            text.set_size(8)

        for i_top, x_col in enumerate(shap_importance.index):
            ax = axes_shap_sca[i_top, i]
            plt.sca(ax)
            x_idx = x_cols.index(x_col)
            plt.scatter(post_df[x_col].values, shap_values[:, x_idx], alpha=0.3, c=scat_c, edgecolors=line_c, s=10, linewidths=.4)
            lowess_result = sm.nonparametric.lowess(shap_values[:, x_idx], post_df[x_col].values, frac=0.35)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=1., alpha=0.7)
            plt.axhline(0, color=line_c, ls='--', lw=0.5, alpha=.8)
            ax.tick_params(labelsize=6, width=.5, length=1, pad=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i_top == 6:
                ax.set_ylabel('SHAP value for {}'.format(y_labels[i]), fontsize=8, labelpad=1)
            if i_top == 0:
                plt.title(f"Target: {y_labels[i]}", fontsize=10, fontweight='bold')
            ax.set_xlabel('Rank {}: {}'.format(i_top + 1, abb1_dic[x_col]), fontsize=7, labelpad=1)
            plt.ylim(ylim_ls[i])

    plt.sca(axes_rf_fitness[0])
    plt.tight_layout()
    plt.savefig('figs/SI 15 RF fitness for del tn.png', dpi=300)

    plt.sca(axes_shap_sum[0])
    plt.tight_layout()
    plt.savefig('figs/SI 16 SHAP summary del tn.png', dpi=300)

    plt.sca(axes_shap_sca[0, 0])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45, wspace=0.4)
    plt.savefig('figs/SI 17 SHAP scatter del tn.png', dpi=300)


def fig_18_tp_shap_compare():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]

    i_study = 9


    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)


    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])


    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values


    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0

    meta_df['Mix_depth_ratio'] = np.clip(meta_df['mixing_depth'] / meta_df['Depth_avg'], 0, 1)
    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Re_wind_exposure_idx'] = meta_df['Wind_exposure_idx'] / meta_df['Depth_avg']
    post_df = post_df.join(meta_df, on='Unique Lake')
    print(post_df.head(), post_df.columns)

    post_df['Dis_avg'] = np.clip(post_df['Dis_avg'], 1, None)
    post_df['Res_time'] = np.clip(post_df['Res_time'], 1, None)
    post_df['Elevation'] = np.clip(post_df['Elevation'], 1, None)
    post_df['Slope_100'] = np.clip(post_df['Slope_100'], 1, None)
    post_df['abs_Pour_lat'] = np.abs(post_df['Pour_lat'])
    post_df['Wshd_lake_ratio'] = post_df['Wshd_area'] / post_df['Lake_area']


    compare_cols = [
        'abs_Pour_lat', 'Elevation',
        'Lake_area', 'Shore_dev', 'Depth_avg', 'Wshd_lake_ratio', 'Vol_total', 'Mix_depth_ratio', 'Wind_exposure_idx',
        'Re_wind_exposure_idx',
        'temp_mean', 'temp_diff', 'temp_sd', 'temp_cv', 'pr_mean', 'pr_diff', 'pr_sd', 'pr_cv', 'ws_mean',
        'ws_diff', 'ws_sd', 'ws_cv', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv',
        'Dis_avg', 'Res_time',
        'TP_mean', 'TP_sd', 'TP_cv',
    ]
    no_log_cols = ['abs_Pour_lat', 'temp_diff', 'temp_mean', 'temp_sd', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv']
    for c in compare_cols:
        if c not in no_log_cols:
            post_df[c] = np.log10(post_df[c])


    x_cols = [
        'temp_mean',
        'Depth_avg',
        'Res_time',
        'TP_mean',
        'Lake_area',
        'Shore_dev'
            ]
    ys = ['alpha', 'beta1', 'beta2', 'sigma']
    y_labels = [r"$\beta_{0}$", r"$\beta_{1}$", r"$\beta_{2}$", r"$\tau$", "R$^2$"]
    fig, axes = plt.subplots(len(ys), 2, figsize=(8, 10))
    for i, y in enumerate(ys):
        rf = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(post_df[x_cols].values, post_df[y].values)
        post_df['TP'] = np.power(10, post_df['TP_mean'])

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(post_df[x_cols])
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.Series(shap_abs_mean, index=x_cols).sort_values(ascending=False)
        most_important_x = shap_importance.index[0]
        post_df['shap_value1'] = shap_values[:, x_cols.index(most_important_x)]

        plt.sca(axes[i, 0])
        plt.scatter(post_df['TP'].values, post_df[y].values,
                    alpha=0.6, c=scat_c, edgecolors=line_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df[y].values, post_df['TP'].values, frac=0.4)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=2)
        plt.axhline(0, color=line_c, ls='--', lw=0.5)
        plt.xscale('log')
        plt.ylabel(y_labels[i])
        plt.xlabel('TP (µg/L)')
        plt.gca().grid(True, which='both', axis='x')

        plt.sca(axes[i, 1])
        plt.scatter(post_df['TP'].values, post_df['shap_value1'].values,
                    alpha=0.6, c=scat_c, edgecolors=line_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df['shap_value1'].values, post_df['TP'].values, frac=0.25)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=2)
        plt.axhline(0, color='#7F8C8D', ls='--', lw=0.5)


        if y in ['beta1', 'beta2']:
            y_values, x_values = post_df['shap_value1'].values, post_df['TP'].values
            x_values = np.log10(x_values)
            y_values = (y_values - y_values.mean()) / y_values.std()

            if y == 'beta1':
                tau1_prior = [np.log10(10), 1]
                delta_prior = [np.log10(30) - np.log10(10), 1]
            else:
                tau1_prior = [np.log10(15), 1]
                delta_prior = [np.log10(40) - np.log10(15), 1]
            with pm.Model() as model1:
                tau1 = pm.Normal("tau1", tau1_prior[0], tau1_prior[1])
                delta = pm.TruncatedNormal("delta", delta_prior[0], delta_prior[1], lower=0.1)
                tau2 = pm.Deterministic("tau2", tau1 + delta)
                alpha1 = pm.Normal("a1", 0, 1)
                beta1 = pm.Normal("b1", 0, 1)
                beta2 = pm.Normal("b2", 0, 1)
                beta3 = pm.Normal("b3", 0, 1)
                c2 = pm.LogNormal("c2", 1, 1)
                sigma = pm.HalfNormal("sigma", 1)

                quad_mid = pm.math.maximum(0, x_values - tau1) ** 2
                mu = alpha1 + beta1 * x_values + (beta2 - beta1) * pm.math.maximum(0, x_values - tau1) + (beta3 - beta2) * pm.math.maximum(0, x_values - tau2)  - c2 * quad_mid + (0 + c2) * pm.math.maximum(0, x_values - tau2) ** 2
                y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_values)
                trace = pm.sample(1000, tune=1000, target_accept=0.98, chains=4, random_seed=911)


            tau1_post = trace.posterior["tau1"].values.flatten()
            tau2_post = trace.posterior["tau2"].values.flatten()
            tau1_low, tau1_median, tau1_high = np.percentile(tau1_post, [2.5, 50, 97.5])
            tau2_low, tau2_median, tau2_high = np.percentile(tau2_post, [2.5, 50, 97.5])
            print(y, 10**tau1_median, 10**tau2_median)
            plt.axvline(10**tau1_median, color=mark_c, ls='-', lw=2)
            plt.axvspan(10**tau1_low, 10**tau1_high, color=mark_c, alpha=0.2, edgecolor='none', linewidth=0)
            plt.axvline(10**tau2_median, color=mark_c, ls='-', lw=2)
            plt.axvspan(10**tau2_low, 10**tau2_high, color=mark_c, alpha=0.2, edgecolor='none', linewidth=0)
            plt.text((10**tau1_median) * 0.9, 0.6, 'Onset{}\nthreshold: {:.0f}'.format(' '*6, 10**tau1_median), transform=plt.gca().get_xaxis_transform(), fontsize=9, va='center', ha='right', color=mark_c, multialignment='right')
            plt.text((10**tau2_median) * 1.1, 0.4, 'Saturation\nthreshold: {:.0f}'.format(10**tau2_median), transform=plt.gca().get_xaxis_transform(), fontsize=9, va='center', ha='left', color=mark_c, multialignment='left')

        plt.xscale('log')
        plt.ylabel('SHAP value' + ' for {}'.format(y_labels[i]))
        plt.xlabel('TP (µg/L)')
        plt.gca().grid(True, which='both', axis='x')


    for ax, label in {
        axes[0, 0]: 'a', axes[0, 1]: 'b',
        axes[1, 0]: 'c', axes[1, 1]: 'd',
        axes[2, 0]: 'e', axes[2, 1]: 'f',
        axes[3, 0]: 'g', axes[3, 1]: 'h',
    }.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('figs/SI 18 SHAP of tp.png', dpi=300)


def fig_18_tp_shap_compare_for_del_tn():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    meta_df = meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)]
    ini_data_df = copy.deepcopy(data_df)


    i_study = 20

    data_df = data_df.dropna(subset=['Chla (µg/L)', 'TN (µg/L)'])
    data_df['TN (µg/L)_std'] = calc_std_tn(data_df['TP (µg/L)'].values)
    print('tn data', len(data_df))
    data_df = data_df[data_df['TN (µg/L)_std'] < data_df['TN (µg/L)'] * 1]
    print('tn with tp limit data', len(data_df))
    data_df['TP (µg/L)_log'] = np.log10(data_df['TP (µg/L)'])
    data_df['Chla (µg/L)_log'] = np.log10(data_df['Chla (µg/L)'])
    lake_counts = data_df.groupby('Unique Lake').size()
    idx = lake_counts.index.intersection(meta_df.index)
    meta_df.loc[idx, 'n_samples_tp_limit'] = lake_counts.loc[idx].values
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[meta_df['n_samples_tp_limit'] > 15].index)]


    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values

    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values


    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0

    meta_df['Mix_depth_ratio'] = np.clip(meta_df['mixing_depth'] / meta_df['Depth_avg'], 0, 1)
    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Re_wind_exposure_idx'] = meta_df['Wind_exposure_idx'] / meta_df['Depth_avg']
    post_df = post_df.join(meta_df, on='Unique Lake')
    print(post_df.head(), post_df.columns)

    post_df['Dis_avg'] = np.clip(post_df['Dis_avg'], 1, None)
    post_df['Res_time'] = np.clip(post_df['Res_time'], 1, None)
    post_df['Elevation'] = np.clip(post_df['Elevation'], 1, None)
    post_df['Slope_100'] = np.clip(post_df['Slope_100'], 1, None)
    post_df['abs_Pour_lat'] = np.abs(post_df['Pour_lat'])
    post_df['Wshd_lake_ratio'] = post_df['Wshd_area'] / post_df['Lake_area']


    compare_cols = [
        'abs_Pour_lat', 'Elevation',
        'Lake_area', 'Shore_dev', 'Depth_avg', 'Wshd_lake_ratio', 'Vol_total', 'Mix_depth_ratio', 'Wind_exposure_idx',
        'Re_wind_exposure_idx',
        'temp_mean', 'temp_diff', 'temp_sd', 'temp_cv', 'pr_mean', 'pr_diff', 'pr_sd', 'pr_cv', 'ws_mean',
        'ws_diff', 'ws_sd', 'ws_cv', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv',
        'Dis_avg', 'Res_time',
        'TP_mean', 'TP_sd', 'TP_cv',
    ]
    no_log_cols = ['abs_Pour_lat', 'temp_diff', 'temp_mean', 'temp_sd', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv']
    for c in compare_cols:
        if c not in no_log_cols:
            post_df[c] = np.log10(post_df[c])


    x_cols = [
        'temp_mean',
        'Depth_avg',
        'Res_time',
        'TP_mean',
        'Lake_area',
        'Shore_dev'
            ]
    ys = ['alpha', 'beta1', 'beta2', 'sigma']
    y_labels = [r"$\beta_{0}$", r"$\beta_{1}$", r"$\beta_{2}$", r"$\tau$", "R$^2$"]
    fig, axes = plt.subplots(len(ys), 2, figsize=(8, 10))
    for i, y in enumerate(ys):
        rf = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(post_df[x_cols].values, post_df[y].values)
        post_df['TP'] = np.power(10, post_df['TP_mean'])

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(post_df[x_cols])
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.Series(shap_abs_mean, index=x_cols).sort_values(ascending=False)
        most_important_x = shap_importance.index[0]
        post_df['shap_value1'] = shap_values[:, x_cols.index(most_important_x)]

        plt.sca(axes[i, 0])
        plt.scatter(post_df['TP'].values, post_df[y].values,
                    alpha=0.6, c=scat_c, edgecolors=line_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df[y].values, post_df['TP'].values, frac=0.35)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=2)
        plt.axhline(0, color=line_c, ls='--', lw=0.5)
        plt.xscale('log')
        plt.ylabel(y_labels[i])
        plt.xlabel('TP (µg/L)')
        plt.gca().grid(True, which='both', axis='x')

        plt.sca(axes[i, 1])
        plt.scatter(post_df['TP'].values, post_df['shap_value1'].values,
                    alpha=0.6, c=scat_c, edgecolors=line_c, s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df['shap_value1'].values, post_df['TP'].values, frac=0.25 if y == 'beta1' else 0.3)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=2)
        plt.axhline(0, color='#7F8C8D', ls='--', lw=0.5)


        if y in ['beta1', 'beta2']:
            y_values, x_values = post_df['shap_value1'].values, post_df['TP'].values
            x_values = np.log10(x_values)
            y_values = (y_values - y_values.mean()) / y_values.std()

            if y == 'beta1':
                tau1_prior = [np.log10(10), 1]
                delta_prior = [np.log10(50) - np.log10(10), 1]
            else:
                tau1_prior = [np.log10(15), 1]
                delta_prior = [np.log10(40) - np.log10(15), 1]
            with pm.Model() as model1:
                tau1 = pm.Normal("tau1", tau1_prior[0], tau1_prior[1])
                delta = pm.TruncatedNormal("delta", delta_prior[0], delta_prior[1], lower=0.1)
                tau2 = pm.Deterministic("tau2", tau1 + delta)
                alpha1 = pm.Normal("a1", 0, 1)
                beta1 = pm.Normal("b1", 0, 1)
                beta2 = pm.Normal("b2", 0, 1)
                beta3 = pm.Normal("b3", 0, 1)
                c2 = pm.LogNormal("c2", 1, 1)
                sigma = pm.HalfNormal("sigma", 1)

                quad_mid = pm.math.maximum(0, x_values - tau1) ** 2
                mu = alpha1 + beta1 * x_values + (beta2 - beta1) * pm.math.maximum(0, x_values - tau1) + (beta3 - beta2) * pm.math.maximum(0, x_values - tau2)  - c2 * quad_mid + (0 + c2) * pm.math.maximum(0, x_values - tau2) ** 2
                y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_values)
                trace = pm.sample(1000, tune=1000, target_accept=0.98, chains=4, random_seed=911)


            tau1_post = trace.posterior["tau1"].values.flatten()
            tau2_post = trace.posterior["tau2"].values.flatten()
            tau1_low, tau1_median, tau1_high = np.percentile(tau1_post, [2.5, 50, 97.5])
            tau2_low, tau2_median, tau2_high = np.percentile(tau2_post, [2.5, 50, 97.5])
            print(y, 10**tau1_median, 10**tau2_median)
            plt.axvline(10**tau1_median, color=mark_c, ls='-', lw=2)
            plt.axvspan(10**tau1_low, 10**tau1_high, color=mark_c, alpha=0.2, edgecolor='none', linewidth=0)
            plt.axvline(10**tau2_median, color=mark_c, ls='-', lw=2)
            plt.axvspan(10**tau2_low, 10**tau2_high, color=mark_c, alpha=0.2, edgecolor='none', linewidth=0)
            plt.text((10**tau1_median) * 0.9, 0.6, 'Onset{}\nthreshold: {:.0f}'.format(' '*6, 10**tau1_median), transform=plt.gca().get_xaxis_transform(), fontsize=9, va='center', ha='right', color=mark_c, multialignment='right')
            plt.text((10**tau2_median) * 1.1, 0.4, 'Saturation\nthreshold: {:.0f}'.format(10**tau2_median), transform=plt.gca().get_xaxis_transform(), fontsize=9, va='center', ha='left', color=mark_c, multialignment='left')

        plt.xscale('log')
        plt.ylabel('SHAP value' + ' for {}'.format(y_labels[i]))
        plt.xlabel('TP (µg/L)')
        plt.gca().grid(True, which='both', axis='x')


    for ax, label in {
        axes[0, 0]: 'a', axes[0, 1]: 'b',
        axes[1, 0]: 'c', axes[1, 1]: 'd',
        axes[2, 0]: 'e', axes[2, 1]: 'f',
        axes[3, 0]: 'g', axes[3, 1]: 'h',
    }.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('figs/SI 18 SHAP of tp del tn.png', dpi=300)


def fig_1920_modeled_P_star_adjust_along_TP():
    def p_star_ad(tp_arr, p_star):
        return p_star * tp_arr / (p_star + tp_arr)

    tp_arr = np.linspace(1, 200, 1000)
    plt.figure(figsize=(5, 4))
    p_star_ls = [20]
    plt.plot(tp_arr, tp_arr, ls='-', lw=2, c='grey', label='1:1 line', alpha=0.3)

    for i, p_star in enumerate(p_star_ls):
        plt.plot(tp_arr, [p_star] * len(tp_arr), ls='--', lw=2, c=fill_c, label='P$^*$ = 20 µg/L')
        plt.plot(tp_arr, p_star_ad(tp_arr, p_star), ls='-', lw=2, c=fill_c, label='Adjusted P$^*$')
        plt.ylabel('Equilibrium P (µg/L)')
        plt.xlabel('TP (µg/L)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('figs/SI 1920 Pstar_adj with TP.png', dpi=300)


def fig_20to22_modeled_chl_under_tp_and_t_smooth():
    data_df, meta_df = get_lake_data(easy_get=True)

    data_df = data_df[
        data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log10_TP'] = np.log10(data_df['TP (µg/L)'])
    data_df['log10_Chla'] = np.log10(data_df['Chla (µg/L)'])

    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)
    plt.figure()
    plt.scatter(mean_df['TP (µg/L)'], mean_df['Chla (µg/L)'], c=mean_df[temp_col], cmap='Reds')
    lowess_result = sm.nonparametric.lowess(mean_df['Chla (µg/L)'], mean_df['TP (µg/L)'], frac=0.2)
    plt.plot(lowess_result[:, 0], lowess_result[:, 1], c='r', ls='--', linewidth=1)
    plt.xscale('log')
    plt.yscale('log')


    fig20, ax_model_heatmap = plt.subplots(1, 1, figsize=(8, 4))
    plt.sca(ax_model_heatmap)
    x = np.linspace(3, 200, 100)
    y = np.linspace(0, 30, 80)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid = np.stack((X, Y), axis=-1).reshape(-1, 2)  # n_x, n_y, 2 -> n_x * n_y, 2

    tp_arr, t_arr = grid[:, 0], grid[:, 1]
    sim_chl = calc_chl(tp_arr, t_arr)
    Z = sim_chl.reshape(len(x), len(y))

    heatmap = plt.pcolormesh(X, Y, Z, cmap=morandi_green, norm=LogNorm(), shading='gouraud')
    cbar = plt.colorbar(heatmap)
    plt.xscale('log')
    plt.xlim(3, 200)
    # plt.ylim(0, 30)
    cbar.ax.set_yscale('log')
    cbar.set_label('Modeled equilibrium Chl$a$ (µg/L)')
    plt.xlabel('TP (µg/L)', labelpad=0)
    plt.ylabel('T (°C)')

    plt.sca(ax_model_heatmap)
    plt.tight_layout()
    plt.savefig('figs/SI 20 Modeled chla smooth.png', dpi=300)


    fig21, axes = plt.subplots(1, 2, figsize=(8, 4))
    plt.sca(axes[0])
    t_slice = [0, 10, 20, 30]
    tp_arr = np.linspace(3, 200, 1000)
    sampled_temp_colors = get_gradient_color_dict(t_slice, cmap=morandi_warm)
    for t in t_slice:
        sim_chl = calc_chl(tp_arr, t)
        plt.plot(tp_arr, sim_chl, c=sampled_temp_colors[t], ls='-', linewidth=2, label='T = {} °C'.format(t))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('TP (µg/L)')
    plt.ylabel('Modeled equilibrium Chl$a$ (µg/L)')
    plt.legend(frameon=False, fontsize=7, loc='upper left')

    plt.sca(axes[1])
    tp_slice = [3, 10, 30, 100]
    t_arr = np.linspace(0, 30, 1000)
    sampled_tp_colors = get_gradient_color_dict(tp_slice, cmap=morandi_green)
    for tp in tp_slice:
        sim_chl = calc_chl(tp, t_arr)
        plt.plot(t_arr, sim_chl, c=sampled_tp_colors[tp], ls='-', linewidth=2, label='TP = {} µg/L'.format(tp))
    plt.yscale('log')
    plt.xlabel('T (°C)')
    plt.ylabel('Modeled equilibrium Chl$a$ (µg/L)')
    plt.legend(frameon=False, fontsize=7, loc='upper left')

    for ax, label in {axes[0]: 'a', axes[1]: 'b'}.items():
        ax.text(0.1, 0.75, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right', zorder=999)

    plt.tight_layout()
    plt.savefig('figs/SI 21 Modeled chla at t and tp.png', dpi=300)


def fig_23_different_min_records():
    min_records = [10, 20, 30, 40, 50]
    n_samples = 2000

    fig, axes = plt.subplots(5, len(min_records), figsize=(8, 9))
    for i_c, min_record in enumerate(min_records):
        data_df, meta_df = get_lake_data(easy_get=True)
        data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= min_record) & (meta_df['n_month'] >= 6)].index)]
        data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
        n_lakes = data_df['lake_idx'].nunique()
        print('used lakes in min {}'.format(min_record), n_lakes)

        temp_col = 'awq_lake_mix_layer_temperature'
        chla = data_df['Chla (µg/L)'].values
        lake_idx = data_df['lake_idx'].values
        data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

        lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
        lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
        data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
        data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
        data_df['temp_centered'] = data_df[temp_col] - 10
        log_TP = data_df['log_TP'].values
        log_TP_centered = data_df['log_TP_centered'].values
        temp_centered = data_df['temp_centered'].values

        with pm.Model() as model:
            alpha_0 = pm.Normal('alpha_0', mu=0, sigma=5)
            beta1_0 = pm.Normal('beta1_0', mu=0, sigma=5)
            beta2_0 = pm.Normal('beta2_0', mu=0, sigma=5)
            # beta3_0 = pm.Normal('beta3_0', mu=0, sigma=5)

            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
            sigma_beta1 = pm.HalfNormal('sigma_beta1', sigma=1)
            sigma_beta2 = pm.HalfNormal('sigma_beta2', sigma=1)
            # sigma_beta3 = pm.HalfNormal('sigma_beta3', sigma=1)
            sigma_sigma = pm.HalfNormal('sigma_sigma', sigma=1)

            alpha = pm.Normal('alpha', mu=alpha_0, sigma=sigma_alpha, shape=n_lakes)
            beta1 = pm.Normal('beta1', mu=beta1_0, sigma=sigma_beta1, shape=n_lakes)
            beta2 = pm.Normal('beta2', mu=beta2_0, sigma=sigma_beta2, shape=n_lakes)
            # beta3 = pm.Normal('beta3', mu=beta3_0, sigma=sigma_beta3, shape=n_lakes)
            sigma = pm.HalfNormal('sigma', sigma=sigma_sigma, shape=n_lakes)

            mu = alpha[lake_idx] + beta1[lake_idx] * log_TP_centered + beta2[lake_idx] * temp_centered
            chla_obs = pm.LogNormal('chla_obs', mu=mu, sigma=sigma[lake_idx], observed=chla)

            trace = pm.sample(draws=n_samples, tune=n_samples, target_accept=0.9, random_seed=911)
            ppc = pm.sample_posterior_predictive(trace)
        az.summary(trace)

        alpha_post = trace.posterior["alpha"].values  # (chain, draw, lake)
        beta1_post = trace.posterior["beta1"].values
        beta2_post = trace.posterior["beta2"].values
        # beta3_post = trace.posterior["beta3"].values
        sigma_post = trace.posterior["sigma"].values

        n_chain, n_draw, _ = alpha_post.shape
        alpha_post = alpha_post.reshape(-1, n_lakes)  # (chain * draw, lake)
        beta1_post = beta1_post.reshape(-1, n_lakes)
        beta2_post = beta2_post.reshape(-1, n_lakes)
        # beta3_post = beta3_post.reshape(-1, n_lakes)
        sigma_post = sigma_post.reshape(-1, n_lakes)

        beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
        beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)
        post_df = pd.DataFrame(index=lake_codes)
        post_df.index.name = 'Unique Lake'
        post_df['beta1'] = beta1_post_median
        post_df['beta2'] = beta2_post_median

        post_df = post_df.join(meta_df, on='Unique Lake')

        pred_ls, mea_ls = [], []
        for lake_id in range(n_lakes):
            lake_mask = lake_idx == lake_id
            tp_i = log_TP_centered[lake_mask]
            temp_i = temp_centered[lake_mask]
            chla_i = chla[lake_mask]
            chla_i = np.log(chla_i)
            # n_draw, n_obs
            mu_samples = ((alpha_post[:, lake_id].reshape(-1, 1)
                           + beta1_post[:, lake_id].reshape(-1, 1) * tp_i.reshape(1, -1))
                          + beta2_post[:, lake_id].reshape(-1, 1) * temp_i.reshape(1, -1))
            mea_ls.extend(chla_i)
            pred_ls.extend(np.mean(mu_samples, axis=0))
        overall_r2 = r2_score(mea_ls, pred_ls)


        plt.sca(axes[0, i_c])
        plt.scatter(np.exp(mea_ls), np.exp(pred_ls), alpha=0.2, c=scat_c, edgecolors=line_c, s=7, linewidths=.1, zorder=10)
        y_max = max(max(np.exp(mea_ls)), max(np.exp(pred_ls)))
        plt.plot([1, y_max], [1, y_max], ls='--', c=line_c, lw=1.5, alpha=0.5, zorder=100)
        plt.text(0.05, 0.95, 'No. of lakes: {}\nOverall R$^2$: {:.2f}'.format(n_lakes, overall_r2), transform=axes[0, i_c].transAxes, fontsize=8, va='top', ha='left', zorder=100)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Measured Chla (µg/L)', fontsize=7, labelpad=1)
        if i_c == 0:
            plt.ylabel('Predicted Chla (µg/L)', fontsize=7, labelpad=1)
        plt.tick_params(labelsize=6, width=.5, which='major', length=1, pad=1)
        plt.tick_params(axis='both', which='minor', length=0)
        plt.xlim(1, 10000)
        plt.ylim(1, 10000)
        plt.title("Min. record: {}".format(min_record), fontsize=9, fontweight='bold')


        ax_cols = ['beta1', 'beta2']
        ax_x_labels = [r'$\beta_1$ dist.', r'$\beta_2$ dist.']
        for i_a, ax in enumerate([axes[1, i_c], axes[2, i_c]]):
            plt.sca(ax)
            data = post_df[ax_cols[i_a]].values
            plt.hist(data, bins=25, density=True, alpha=0.8, color='#BDC3C7', edgecolor='white')

            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            plt.plot(x_range, kde(x_range), color=line_c, lw=1.5, alpha=1)
            plt.axvline(0, color=line_c, linestyle='--', lw=0.5, alpha=0.6)
            plt.xlabel(ax_x_labels[i_a], fontsize=7, labelpad=1)
            if i_c == 0:
                plt.ylabel("Density", fontsize=7, labelpad=1)
            plt.tick_params(labelsize=6, width=.5, which='major', length=1, pad=1)
            plt.tick_params(axis='both', which='minor', length=0)


        ys = ['beta1', 'beta2']
        y_label_dic = {'beta1': r"$\beta_{1}$", 'beta2': r"$\beta_{2}$"}
        for i_a, ax in enumerate([axes[3, i_c], axes[4, i_c]]):
            y = ys[i_a]
            plt.sca(ax)
            plt.scatter(post_df['TP_mean'].values, post_df[y].values, alpha=0.5, c=scat_c, edgecolors=line_c, s=10, linewidths=.1)
            lowess_result = sm.nonparametric.lowess(post_df[y].values, post_df['TP_mean'].values, frac=0.5)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=1.5)
            plt.axhline(0, color=line_c, linestyle='--', lw=0.5, alpha=0.6)
            plt.xscale('log')
            plt.xlabel('Lake mean TP (µg/L)', fontsize=7, labelpad=1)
            if i_c == 0:
                plt.ylabel(y_label_dic[y], fontsize=7, labelpad=1)
            plt.tick_params(labelsize=6, width=.5, which='major', length=1, pad=1)
            plt.tick_params(axis='both', which='minor', length=0)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig('figs/SI 23 Different min records.png', dpi=300)


def fig_24_different_min_months():
    min_months = [2, 4, 6, 8, 10]
    n_samples = 2000

    fig, axes = plt.subplots(5, len(min_months), figsize=(8, 9))
    for i_c, min_month in enumerate(min_months):
        data_df, meta_df = get_lake_data(easy_get=True)
        data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= min_month)].index)]
        data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
        n_lakes = data_df['lake_idx'].nunique()
        print('used lakes in min month {}'.format(min_month), n_lakes)

        temp_col = 'awq_lake_mix_layer_temperature'
        chla = data_df['Chla (µg/L)'].values
        lake_idx = data_df['lake_idx'].values
        data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

        lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
        lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
        data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
        data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
        data_df['temp_centered'] = data_df[temp_col] - 10
        log_TP = data_df['log_TP'].values
        log_TP_centered = data_df['log_TP_centered'].values
        temp_centered = data_df['temp_centered'].values

        with pm.Model() as model:
            alpha_0 = pm.Normal('alpha_0', mu=0, sigma=5)
            beta1_0 = pm.Normal('beta1_0', mu=0, sigma=5)
            beta2_0 = pm.Normal('beta2_0', mu=0, sigma=5)
            # beta3_0 = pm.Normal('beta3_0', mu=0, sigma=5)

            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
            sigma_beta1 = pm.HalfNormal('sigma_beta1', sigma=1)
            sigma_beta2 = pm.HalfNormal('sigma_beta2', sigma=1)
            # sigma_beta3 = pm.HalfNormal('sigma_beta3', sigma=1)
            sigma_sigma = pm.HalfNormal('sigma_sigma', sigma=1)

            alpha = pm.Normal('alpha', mu=alpha_0, sigma=sigma_alpha, shape=n_lakes)
            beta1 = pm.Normal('beta1', mu=beta1_0, sigma=sigma_beta1, shape=n_lakes)
            beta2 = pm.Normal('beta2', mu=beta2_0, sigma=sigma_beta2, shape=n_lakes)
            # beta3 = pm.Normal('beta3', mu=beta3_0, sigma=sigma_beta3, shape=n_lakes)
            sigma = pm.HalfNormal('sigma', sigma=sigma_sigma, shape=n_lakes)

            mu = alpha[lake_idx] + beta1[lake_idx] * log_TP_centered + beta2[lake_idx] * temp_centered
            chla_obs = pm.LogNormal('chla_obs', mu=mu, sigma=sigma[lake_idx], observed=chla)

            trace = pm.sample(draws=n_samples, tune=n_samples, target_accept=0.9, random_seed=911)
            ppc = pm.sample_posterior_predictive(trace)
        az.summary(trace)

        alpha_post = trace.posterior["alpha"].values  # (chain, draw, lake)
        beta1_post = trace.posterior["beta1"].values
        beta2_post = trace.posterior["beta2"].values
        # beta3_post = trace.posterior["beta3"].values
        sigma_post = trace.posterior["sigma"].values

        n_chain, n_draw, _ = alpha_post.shape
        alpha_post = alpha_post.reshape(-1, n_lakes)  # (chain * draw, lake)
        beta1_post = beta1_post.reshape(-1, n_lakes)
        beta2_post = beta2_post.reshape(-1, n_lakes)
        # beta3_post = beta3_post.reshape(-1, n_lakes)
        sigma_post = sigma_post.reshape(-1, n_lakes)

        beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
        beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)
        post_df = pd.DataFrame(index=lake_codes)
        post_df.index.name = 'Unique Lake'
        post_df['beta1'] = beta1_post_median
        post_df['beta2'] = beta2_post_median

        post_df = post_df.join(meta_df, on='Unique Lake')

        pred_ls, mea_ls = [], []
        for lake_id in range(n_lakes):
            lake_mask = lake_idx == lake_id
            tp_i = log_TP_centered[lake_mask]
            temp_i = temp_centered[lake_mask]
            chla_i = chla[lake_mask]
            chla_i = np.log(chla_i)
            # n_draw, n_obs
            mu_samples = ((alpha_post[:, lake_id].reshape(-1, 1)
                           + beta1_post[:, lake_id].reshape(-1, 1) * tp_i.reshape(1, -1))
                          + beta2_post[:, lake_id].reshape(-1, 1) * temp_i.reshape(1, -1))
            mea_ls.extend(chla_i)
            pred_ls.extend(np.mean(mu_samples, axis=0))
        overall_r2 = r2_score(mea_ls, pred_ls)

        plt.sca(axes[0, i_c])
        plt.scatter(np.exp(mea_ls), np.exp(pred_ls), alpha=0.2, c=scat_c, edgecolors=line_c, s=7, linewidths=.1,
                    zorder=10)
        y_max = max(max(np.exp(mea_ls)), max(np.exp(pred_ls)))
        plt.plot([1, y_max], [1, y_max], ls='--', c=line_c, lw=1.5, alpha=0.5, zorder=100)
        plt.text(0.05, 0.95, 'No. of lakes: {}\nOverall R$^2$: {:.2f}'.format(n_lakes, overall_r2),
                 transform=axes[0, i_c].transAxes, fontsize=8, va='top', ha='left', zorder=100)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Measured Chla (µg/L)', fontsize=7, labelpad=1)
        if i_c == 0:
            plt.ylabel('Predicted Chla (µg/L)', fontsize=7, labelpad=1)
        plt.tick_params(labelsize=6, width=.5, which='major', length=1, pad=1)
        plt.tick_params(axis='both', which='minor', length=0)
        plt.xlim(1, 10000)
        plt.ylim(1, 10000)
        plt.title("Min. months: {}".format(min_month), fontsize=9, fontweight='bold')

        ax_cols = ['beta1', 'beta2']
        ax_x_labels = [r'$\beta_1$ dist.', r'$\beta_2$ dist.']
        for i_a, ax in enumerate([axes[1, i_c], axes[2, i_c]]):
            plt.sca(ax)
            data = post_df[ax_cols[i_a]].values
            plt.hist(data, bins=25, density=True, alpha=0.8, color='#BDC3C7', edgecolor='white')

            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            plt.plot(x_range, kde(x_range), color=line_c, lw=1.5, alpha=1)
            plt.axvline(0, color=line_c, linestyle='--', lw=0.5, alpha=0.6)
            plt.xlabel(ax_x_labels[i_a], fontsize=7, labelpad=1)
            if i_c == 0:
                plt.ylabel("Density", fontsize=7, labelpad=1)
            plt.tick_params(labelsize=6, width=.5, which='major', length=1, pad=1)
            plt.tick_params(axis='both', which='minor', length=0)

        ys = ['beta1', 'beta2']
        y_label_dic = {'beta1': r"$\beta_{1}$", 'beta2': r"$\beta_{2}$"}
        for i_a, ax in enumerate([axes[3, i_c], axes[4, i_c]]):
            y = ys[i_a]
            plt.sca(ax)
            plt.scatter(post_df['TP_mean'].values, post_df[y].values, alpha=0.5, c=scat_c, edgecolors=line_c, s=10,
                        linewidths=.1)
            lowess_result = sm.nonparametric.lowess(post_df[y].values, post_df['TP_mean'].values, frac=0.5)
            plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=1.5)
            plt.axhline(0, color=line_c, linestyle='--', lw=0.5, alpha=0.6)
            plt.xscale('log')
            plt.xlabel('Lake mean TP (µg/L)', fontsize=7, labelpad=1)
            if i_c == 0:
                plt.ylabel(y_label_dic[y], fontsize=7, labelpad=1)
            plt.tick_params(labelsize=6, width=.5, which='major', length=1, pad=1)
            plt.tick_params(axis='both', which='minor', length=0)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig('figs/SI 24 Different min months.png', dpi=300)


def fig_25to27_compare_with_interaction_term():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    fig_paras_dist, axes_paras_dist = plt.subplots(5, 2, figsize=(8, 10))
    fig_fitness, axes_fitness = plt.subplots(2, 2, figsize=(8, 8))
    i_studys = [9, 8]
    post_df_dic = {}
    titles = ['Without interaction term', 'With interaction term']
    for i_c, i_study in enumerate(i_studys):
        alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
        beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
        beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
        beta3_post = pd.read_csv('results/samples/beta3_post{}.csv'.format(i_study), header=0, index_col=0).values
        sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values

        alpha_post_low, alpha_post_median, alpha_post_high = np.percentile(alpha_post, [5, 50, 95], axis=0)
        beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
        beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)
        beta3_post_low, beta3_post_median, beta3_post_high = np.percentile(beta3_post, [5, 50, 95], axis=0)
        sigma_post_low, sigma_post_median, sigma_post_high = np.percentile(sigma_post, [5, 50, 95], axis=0)

        post_df = pd.DataFrame(index=lake_codes)
        post_df.index.name = 'Unique Lake'
        post_df['alpha'] = alpha_post_median
        post_df['beta1'] = beta1_post_median
        post_df['beta2'] = beta2_post_median
        post_df['beta3'] = beta3_post_median
        post_df['sigma'] = sigma_post_median

        post_df['beta1_sig_pos'] = beta1_post_low > 0
        post_df['beta2_sig_pos'] = beta2_post_low > 0
        post_df['beta3_sig_pos'] = beta3_post_low > 0
        post_df['beta1_sig_neg'] = beta1_post_high < 0
        post_df['beta2_sig_neg'] = beta2_post_high < 0
        post_df['beta3_sig_neg'] = beta3_post_high < 0

        r2_df = pd.DataFrame(index=lake_codes)
        r2_df.index.name = 'Unique Lake'
        pred_ls, mea_ls = [], []
        for lake_id in range(n_lakes):
            print(lake_id, np.percentile(beta1_post[:, lake_id], [5, 50, 95]),
                  np.percentile(beta2_post[:, lake_id], [5, 50, 95]),
                  np.percentile(sigma_post[:, lake_id], [5, 50, 95]))
            lake_mask = lake_idx == lake_id
            tp_i = log_TP_centered[lake_mask]
            temp_i = temp_centered[lake_mask]
            chla_i = chla[lake_mask]
            chla_i = np.log(chla_i)
            if i_study == 9:
                mu_samples = ((alpha_post[:, lake_id].reshape(-1, 1)
                               + beta1_post[:, lake_id].reshape(-1, 1) * tp_i.reshape(1, -1))
                              + beta2_post[:, lake_id].reshape(-1, 1) * temp_i.reshape(1, -1))
            else:
                mu_samples = alpha_post[:, lake_id].reshape(-1, 1) + beta1_post[:, lake_id].reshape(-1, 1) * tp_i.reshape(1, -1) + beta2_post[:, lake_id].reshape(-1, 1) * temp_i.reshape(1, -1) + beta3_post[:, lake_id].reshape(-1, 1) * tp_i.reshape(1, -1) * temp_i.reshape(1, -1)
            r2 = r2_score(chla_i, np.mean(mu_samples, axis=0))
            r2_df.loc[lake_codes[lake_id], 'r2'] = r2
            mea_ls.extend(np.exp(chla_i))
            pred_ls.extend(np.exp(np.mean(mu_samples, axis=0)))
        post_df = post_df.join(r2_df, on='Unique Lake')
        post_df_dic[i_study] = post_df


        ax_cols = ['alpha', 'beta1', 'beta2', 'sigma', 'beta3']
        ax_inset_title_dic = {'beta1': r"Sig. of $\beta_{1,i}$", 'beta2': r"Sig. of $\beta_{2,i}$", 'beta3': r"Sig. of $\beta_{3,i}$"}
        ax_pie_colors = ['#D98880', '#85929E', '#ECF0F1']
        ax_x_labels = [
            r'Posterior distribution of hyperparameter $\beta_0$',
            r'Posterior distribution of hyperparameter $\beta_1$',
            r'Posterior distribution of hyperparameter $\beta_2$',
            r'Posterior distribution of hyperparameter $\tau$',
            r'Posterior distribution of hyperparameter $\beta_3$']
        for i_r, col in enumerate(ax_cols):
            ax = axes_paras_dist[i_r, i_c]
            if i_study == 9 and col == 'beta3':
                plt.delaxes(ax)
            else:
                plt.sca(ax)
                data = post_df[col].values
                plt.hist(data, bins=25, density=True, alpha=0.6, color='#BDC3C7', edgecolor='white')
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                plt.plot(x_range, kde(x_range), color='#1C2833', lw=2, alpha=1)
                plt.axvline(0, color='#7F8C8D', linestyle='--', lw=0.5, alpha=1)
                if col[:4] == 'beta':
                    ax_inset = ax.inset_axes([0.6, 0.3, 0.4, 0.5])
                    labels = ['+', '—', 'n.s.']
                    counts = [len(post_df[post_df['{}_sig_pos'.format(col)] > 0]), len(post_df[post_df['{}_sig_neg'.format(col)] > 0])]
                    counts.append(n_lakes - counts[0] - counts[1])
                    ax_inset.pie(counts, labels=labels, colors=ax_pie_colors, autopct=lambda p: f'{p:.0f}%' if p > 5 else '', textprops={'fontsize': 7}, startangle=90)
                    ax_inset.set_title(ax_inset_title_dic[col], fontsize=8)
                if i_r == 0:
                    plt.title(titles[i_c], fontsize=10, fontweight='bold')
                plt.xlabel(ax_x_labels[i_r])
                if i_c == 0 or col == 'beta3':
                    plt.ylabel("Density")


        plt.sca(axes_fitness[0, i_c])
        overall_r2 = r2_score(np.log(mea_ls), np.log(pred_ls))
        plt.scatter(mea_ls, pred_ls, alpha=0.3, c=scat_c, edgecolors=line_c, s=20, linewidths=.25, zorder=10)
        plt.text(0.05, 0.9, 'Overall R$^2$: {:.2f}'.format(overall_r2), transform=plt.gca().transAxes, fontsize=10,
                 va='top', ha='left', zorder=100)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Measured Chla (µg/L)')
        plt.ylabel('Predicted mean Chla (µg/L)')
        y_max = max(max(mea_ls), max(pred_ls))
        plt.plot([1, y_max], [1, y_max], ls='--', c=line_c, lw=2, alpha=0.5, zorder=100)
        plt.title(titles[i_c], fontsize=10, fontweight='bold')

        plt.sca(axes_fitness[1, i_c])
        plt.hist(r2_df['r2'].values, bins=20, density=True, color=fill_c, alpha=0.7, edgecolor=line_c, linewidth=0.5)
        print(np.mean(r2_df['r2'].values))
        plt.xlabel('Intra-lake R$^2$')
        plt.ylabel('Density')

    plt.sca(axes_paras_dist[0, 0])
    plt.tight_layout()
    plt.savefig('figs/SI 25 Paras dist diff.png', dpi=300)

    plt.sca(axes_fitness[0, 0])
    plt.tight_layout()
    plt.subplots_adjust(hspace=.3, wspace=.3)
    plt.savefig('figs/SI 27 Fitness diff.png', dpi=300)


    fig_paras_compare, axes_paras_compare = plt.subplots(2, 2, figsize=(8, 8))
    axes_paras_compare = axes_paras_compare.flatten()
    ax_cols = ['alpha', 'beta1', 'beta2', 'sigma']
    ax_labels = [r'$\beta_0$', r'$\beta_1$', r'$\beta_2$', r'$\tau$']
    x_post_df, y_post_df = post_df_dic[9], post_df_dic[8]
    for i_ax, col in enumerate(ax_cols):
        plt.sca(axes_paras_compare[i_ax])
        x, y = x_post_df[col].values, y_post_df[col].values
        plt.scatter(x, y, alpha=0.6, c=scat_c, edgecolors=line_c, s=30, linewidths=.5)
        min_value, max_value = min(min(x), min(y)), max(max(x), max(y))
        plt.plot([min_value, max_value], [min_value, max_value], ls='--', c=line_c, lw=2, alpha=0.5, zorder=99)
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)
        plt.xlabel('Estimated {} (Without interaction term)'.format(ax_labels[i_ax]))
        plt.ylabel('Estimated {} (With interaction term)'.format(ax_labels[i_ax]))
    plt.tight_layout()
    plt.subplots_adjust(hspace=.3, wspace=.3)
    plt.savefig('figs/SI 26 Paras compare.png', dpi=300)


def fig28_depth_slice():
    data_df, meta_df = get_lake_data(easy_get=True)

    data_df = data_df[
        data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    print(data_df.columns)

    i_study = 9

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TP'] = np.log(data_df['TP (µg/L)'])

    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']
    data_df['temp_centered'] = data_df[temp_col] - 10
    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta3_post = pd.read_csv('results/samples/beta3_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values

    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)
    n_beta1_sig_pos, n_beta1_pos, n_beta1_sig_neg = (beta1_post_low > 0).sum(), (beta1_post_median > 0).sum(), (
            beta1_post_high < 0).sum()
    n_beta2_sig_pos, n_beta2_pos, n_beta2_sig_neg = (beta2_post_low > 0).sum(), (beta2_post_median > 0).sum(), (
            beta2_post_high < 0).sum()
    print('beta1 {} pos, {} sig pos, {} sig neg'.format(n_beta1_pos / n_lakes, n_beta1_sig_pos / n_lakes,
                                                        n_beta1_sig_neg / n_lakes))
    print('beta2 {} pos, {} sig pos, {} sig neg'.format(n_beta2_pos / n_lakes, n_beta2_sig_pos / n_lakes,
                                                        n_beta2_sig_neg / n_lakes))
    beta1_post_sd = np.std(beta1_post, axis=0)
    beta2_post_sd = np.std(beta2_post, axis=0)

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['beta1_sd'] = beta1_post_sd
    post_df['beta2_sd'] = beta2_post_sd
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0

    meta_df['Mix_depth_ratio'] = np.clip(meta_df['mixing_depth'] / meta_df['Depth_avg'], 0, 1)
    meta_df['Wind_exposure_idx'] = (meta_df['ws_mean'] ** 2) * np.sqrt(meta_df['Lake_area'])
    meta_df['Re_wind_exposure_idx'] = meta_df['Wind_exposure_idx'] / meta_df['Depth_avg']
    post_df = post_df.join(meta_df, on='Unique Lake')
    post_df['Dynamic_ratio'] = np.sqrt(meta_df['Lake_area']) / meta_df['Depth_avg']
    print(post_df.head(), post_df.columns)

    post_df['Dis_avg'] = np.clip(post_df['Dis_avg'], 1, None)
    post_df['Res_time'] = np.clip(post_df['Res_time'], 1, None)
    post_df['Elevation'] = np.clip(post_df['Elevation'], 1, None)
    post_df['Slope_100'] = np.clip(post_df['Slope_100'], 1, None)
    post_df['abs_Pour_lat'] = np.abs(post_df['Pour_lat'])
    post_df['Wshd_lake_ratio'] = post_df['Wshd_area'] / post_df['Lake_area']

    compare_cols = [
        'abs_Pour_lat', 'Elevation',
        'Lake_area', 'Shore_dev', 'Depth_avg', 'Wshd_lake_ratio', 'Vol_total', 'Mix_depth_ratio', 'Wind_exposure_idx',
        'Re_wind_exposure_idx', 'Dynamic_ratio',
        'temp_mean', 'temp_diff', 'temp_sd', 'temp_cv', 'pr_mean', 'pr_diff', 'pr_sd', 'pr_cv', 'ws_mean',
        'ws_diff', 'ws_sd', 'ws_cv', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv',
        'Dis_avg', 'Res_time',
        'TP_mean', 'TP_sd', 'TP_cv',
    ]
    no_log_cols = ['abs_Pour_lat', 'temp_diff', 'temp_sd', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv']
    for c in compare_cols:
        if c not in no_log_cols:
            post_df[c] = np.log10(post_df[c])
    post_df['TP'] = 10 ** post_df['TP_mean']


    x_cols = [
        'temp_mean',
        'Depth_avg',
        'Res_time',
        'TP_mean',
        'Lake_area',
        'Shore_dev'
            ]
    slice_dic = {
        'Depth_avg': [[0.01, 3], [2, 5], [4, 7], [6, 20], [8, 200]],
                 }
    ys = ['beta1', 'beta2']
    title_dic = {'Depth_avg': 'Depth'}
    y_label_dic = {'beta1': r"$\beta_{1}$", 'beta2': r"$\beta_{2}$", 'sigma': r"$\sigma_{i}$"}

    for slice_col, slice_ls in slice_dic.items():
        fig, axes = plt.subplots(len(ys) * 2, len(slice_ls), figsize=(8, 8), sharey='row')
        for i_y, y in enumerate(ys):
            for j, slice in enumerate(slice_ls):
                left, right = slice
                plt.sca(axes[i_y, j])
                sub_df = post_df[(post_df[slice_col] >= np.log10(left)) & (post_df[slice_col] <= np.log10(right))]
                print(slice_col, slice, len(sub_df))
                plt.scatter(sub_df['TP'].values, sub_df[y].values, alpha=0.5, c=scat_c, edgecolors=line_c, s=10, linewidths=.1)
                lowess_result = sm.nonparametric.lowess(sub_df[y].values, sub_df['TP'].values, frac=0.65)
                plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=1.5)
                if j == 0:
                    plt.ylabel(y_label_dic[y], fontsize=7, labelpad=1)
                plt.xlabel('Lake mean TP (µg/L)', fontsize=7, labelpad=1)
                plt.xscale('log')
                plt.xlim(3, 200)
                plt.tick_params(labelsize=6, width=.5, which='major', length=1, pad=1)
                plt.tick_params(axis='both', which='minor', length=0)
                if i_y == 0:
                    plt.title('{}: {:.0f}-{:.0f} m\n(n={})'.format(title_dic[slice_col], left, right, len(sub_df)), fontsize=9, fontweight='bold')

            for j, slice in enumerate(slice_ls):
                left, right = slice
                plt.sca(axes[i_y+len(ys), j])
                sub_df = post_df[(post_df[slice_col] >= np.log10(left)) & (post_df[slice_col] <= np.log10(right))]
                rf = RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=2,
                    min_samples_split=4,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(sub_df[x_cols].values, sub_df[y].values)
                r2_base = r2_score(sub_df[y].values, rf.predict(sub_df[x_cols].values))
                rf_importance = pd.Series(rf.feature_importances_, index=x_cols)
                print(y, r2_base)
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(sub_df[x_cols])
                shap_abs_mean = np.abs(shap_values).mean(axis=0)
                shap_importance = pd.Series(shap_abs_mean, index=x_cols)
                shap_importance = shap_importance.sort_values(ascending=False)
                # shap_importance = shap_importance.reset_index()
                most_important_x = shap_importance.index[0]
                print(slice, most_important_x)
                sub_df['shap_value1'] = shap_values[:, x_cols.index(most_important_x)]
                plt.scatter(sub_df['TP'].values, sub_df['shap_value1'].values,
                            alpha=0.5, c=scat_c, edgecolors=line_c, s=10, linewidths=.1)
                lowess_result = sm.nonparametric.lowess(sub_df['shap_value1'].values, sub_df['TP'].values, frac=0.65)
                plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=1.5)
                if j == 0:
                    plt.ylabel('SHAP value for {}'.format(y_label_dic[y]), fontsize=7, labelpad=1)
                plt.xlabel('Lake mean TP (µg/L)', fontsize=7, labelpad=1)
                plt.xlim(3, 200)
                plt.xscale('log')
                plt.tick_params(labelsize=6, width=.5, which='major', length=1, pad=1)
                plt.tick_params(axis='both', which='minor', length=0)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.25, hspace=0.5)
        plt.savefig('figs/SI 28 slice in {}.png'.format(slice_col), dpi=300)


def fig29to30_nitrogen_relation():
    data_df, meta_df = get_lake_data(wq_cols=['Chla (µg/L)', 'TN (µg/L)'], min_wq=1, max_wq=6000, easy_get=True)

    i_study = 101


    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]


    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)

    temp_col = 'awq_lake_mix_layer_temperature'
    chla = data_df['Chla (µg/L)'].values
    lake_idx = data_df['lake_idx'].values
    data_df['log_TN'] = np.log(data_df['TN (µg/L)'])

    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)
    lake_means = data_df.groupby('lake_idx')[['log_TN']].mean()
    lake_means = lake_means.rename(columns={'log_TN': 'mean_log_TN'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TN_centered'] = data_df['log_TN'] - data_df['mean_log_TN']

    lake_means = data_df.groupby('lake_idx')[[temp_col]].mean()
    lake_means = lake_means.rename(columns={temp_col: 'mean_{}'.format(temp_col)})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['temp_centered'] = data_df[temp_col] - data_df['mean_{}'.format(temp_col)]

    log_TN = data_df['log_TN'].values
    log_TN_centered = data_df['log_TN_centered'].values
    temp_centered = data_df['temp_centered'].values
    temp_no_centered = data_df[temp_col].values


    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta3_post = pd.read_csv('results/samples/beta3_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values

    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)
    # beta3_post_low, beta3_post_median, beta3_post_high = np.percentile(beta3_post, [5, 50, 95], axis=0)
    n_beta1_sig_pos, n_beta1_pos, n_beta1_sig_neg = (beta1_post_low > 0).sum(), (beta1_post_median > 0).sum(), (
            beta1_post_high < 0).sum()
    n_beta2_sig_pos, n_beta2_pos, n_beta2_sig_neg = (beta2_post_low > 0).sum(), (beta2_post_median > 0).sum(), (
            beta2_post_high < 0).sum()
    # n_beta3_sig_pos, n_beta3_pos, n_beta3_sig_neg = (beta3_post_low > 0).sum(), (beta3_post_median > 0).sum(), (beta3_post_high < 0).sum()
    print('beta1 {} pos, {} sig pos, {} sig neg'.format(n_beta1_pos / n_lakes, n_beta1_sig_pos / n_lakes,
                                                        n_beta1_sig_neg / n_lakes))
    print('beta2 {} pos, {} sig pos, {} sig neg'.format(n_beta2_pos / n_lakes, n_beta2_sig_pos / n_lakes,
                                                        n_beta2_sig_neg / n_lakes))
    # print('beta3 {} pos, {} sig pos, {} sig neg'.format(n_beta3_pos / n_lakes, n_beta3_sig_pos / n_lakes, n_beta3_sig_neg / n_lakes))

    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    # post_df['beta3'] = beta3_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta1_sig_neg'] = beta1_post_high < 0
    post_df['beta2_sig_neg'] = beta2_post_high < 0
    post_df['beta1_no_sig'] = np.where((post_df['beta1_sig_pos'] == 0) & (post_df['beta1_sig_neg'] == 0), 1, 0)
    post_df['beta2_no_sig'] = np.where((post_df['beta2_sig_pos'] == 0) & (post_df['beta2_sig_neg'] == 0), 1, 0)

    post_df['beta1_trend'] = post_df['beta1_sig_pos'].astype(int) - post_df['beta1_sig_neg'].astype(int)
    post_df['beta2_trend'] = post_df['beta2_sig_pos'].astype(int) - post_df['beta2_sig_neg'].astype(int)
    counts = post_df.groupby(['beta1_trend', 'beta2_trend']).size().reset_index(name='Count')
    all_combinations = pd.MultiIndex.from_product([[-1, 0, 1], [-1, 0, 1]], names=['beta1_trend', 'beta2_trend'])
    counts = counts.set_index(['beta1_trend', 'beta2_trend']).reindex(all_combinations, fill_value=0).reset_index()

    post_df['G1_PP'] = (post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G2_PN'] = (post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    post_df['G3_NP'] = (~post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G4_NN'] = (~post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    conditions = [post_df['G1_PP'], post_df['G2_PN'], post_df['G3_NP'], post_df['G4_NN']]
    labels = ['PP', 'PN', 'NP', 'NN']
    post_df['response_type'] = np.select(conditions, labels, default='UNCLASSIFIED')
    lake_counts = post_df.groupby('response_type').size()
    print(lake_counts)

    def classify(row):
        if row['beta1_sig_pos'] and row['beta2_sig_pos']:
            return 'PP'
        elif row['beta1_sig_pos'] and not row['beta2_sig_pos']:
            return 'PN'
        elif not row['beta1_sig_pos'] and row['beta2_sig_pos']:
            return 'NP'
        else:
            return 'NN'

    post_df['Class'] = post_df.apply(classify, axis=1)
    class_map = {'NN': 0, 'NP': 1, 'PN': 2, 'PP': 3}
    post_df['Class_num'] = post_df['Class'].map(class_map).astype(int)

    fig = plt.figure(figsize=(8, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 6], width_ratios=[4, 4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax_cols = ['beta1', 'beta2']
    ax_titles = [r'TN coefficient ($\beta_1$)', r'T coefficient ($\beta_2$)']
    ax_inset_titles = [r"Sig. of $\beta_{1,i}$", r"Sig. of $\beta_{2,i}$"]
    ax_pie_colors = [['#D98880', '#85929E', '#ECF0F1'], ['#A2D9CE', '#85929E', '#ECF0F1']]
    ax_x_labels = [r'Posterior distribution of hyperparameter $\beta_1$',
                   r'Posterior distribution of hyperparameter $\beta_2$']
    for i_a, ax in enumerate([ax1, ax2]):
        plt.sca(ax)
        data = post_df[ax_cols[i_a]].values

        plt.hist(data, bins=20, density=True, alpha=0.6, color='#BDC3C7', edgecolor='white')

        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        plt.plot(x_range, kde(x_range), color='#1C2833', lw=2, alpha=1)

        plt.axvline(0, color='#7F8C8D', linestyle='--', lw=0.5, alpha=1)

        ax_inset = ax.inset_axes([0.6, 0.5, 0.4, 0.4])
        labels = ['+', '—', 'n.s.']
        g_counts = [len(post_df[post_df['{}_sig_pos'.format(ax_cols[i_a])] > 0]),
                    len(post_df[post_df['{}_sig_neg'.format(ax_cols[i_a])] > 0])]
        g_counts.append(n_lakes - g_counts[0] - g_counts[1])
        ax_inset.pie(g_counts, labels=labels, colors=ax_pie_colors[i_a], autopct='%1.0f%%', textprops={'fontsize': 7},
                     startangle=90)
        ax_inset.set_title(ax_inset_titles[i_a], fontsize=8)

        plt.title(ax_titles[i_a])
        plt.xlabel(ax_x_labels[i_a])
        if i_a == 0:
            plt.ylabel("Density")

    plt.sca(ax3)
    tick_labels = ['—', 'n.s.', '+']
    tick_values = [-1, 0, 1]
    bubble_scale = 30
    print(counts)
    c9_ls = ['#85929e', '#ecf0f1', '#a2d9ce',
             '#ecf0f1', '#ecf0f1', '#a2d9ce',
             '#d98880', '#d98880', '#b03a2e']
    scatter = plt.scatter(
        counts['beta1_trend'],
        counts['beta2_trend'],
        s=counts['Count'] * bubble_scale,
        c=c9_ls,
        alpha=1,
        edgecolors='none',
        linewidths=0.6,
        zorder=2
    )
    for i, row in counts.iterrows():
        if row['Count'] > -1:
            plt.text(
                row['beta1_trend'],
                row['beta2_trend'],
                f"n={int(row['Count'])}",
                ha='center',
                va='center',
                fontsize=9,
                zorder=100
            )

    plt.plot([0.55, 1.45, 1.45, 0.55, 0.55], [0.55, 0.55, 1.45, 1.45, 0.55], c=c9_ls[8], lw=2, ls='-')
    plt.text(1.4, 1.4, 'PP', fontsize=12, va='top', ha='right')
    plt.plot([0.55, 1.45, 1.45, 0.55, 0.55], [-1.45, -1.45, 0.45, 0.45, -1.45], c=c9_ls[7], lw=2, ls='-')
    plt.text(1.4, 0.4, 'PN', fontsize=12, va='top', ha='right')
    plt.plot([-1.25, 0.45, 0.45, -1.25, -1.25], [0.55, 0.55, 1.45, 1.45, 0.55], c=c9_ls[2], lw=2, ls='-')
    plt.text(0.4, 1.4, 'NP', fontsize=12, va='top', ha='right')
    plt.plot([-1.25, 0.45, 0.45, -1.25, -1.25], [-1.45, -1.45, 0.45, 0.45, -1.45], c=c9_ls[1], lw=2, ls='-')
    plt.text(0.4, 0.4, 'NN', fontsize=12, va='top', ha='right')

    plt.xticks(tick_values, tick_labels, fontsize=8)
    plt.yticks(tick_values, tick_labels, fontsize=8)
    plt.xlim(-1.6, 1.6)
    plt.ylim(-1.6, 1.6)
    plt.xlabel(ax_inset_titles[0], fontsize=9)
    plt.ylabel(ax_inset_titles[1], fontsize=9)

    for ax, label in {ax1: 'a', ax2: 'b', ax3: 'c'}.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('figs/SI 29 paras class - TN.png', dpi=300)


    meta_df['Mix_depth_ratio'] = np.clip(meta_df['mixing_depth'] / meta_df['Depth_avg'], 0, 1)
    meta_df['Wind_exposure_idx'] = meta_df['ws_mean'] * np.sqrt(meta_df['Lake_area'])
    meta_df['Re_wind_exposure_idx'] = meta_df['Wind_exposure_idx'] / meta_df['Depth_avg']
    post_df = post_df.join(meta_df, on='Unique Lake')
    post_df['abs_Pour_lat'] = np.abs(post_df['Pour_lat'])
    post_df['Wshd_lake_ratio'] = post_df['Wshd_area'] / post_df['Lake_area']
    print(post_df.head(), post_df.columns)

    compare_cols = [
        'abs_Pour_lat', 'Elevation',
        'Lake_area', 'Shore_dev', 'Depth_avg', 'Vol_total', 'Mix_depth_ratio', 'Wind_exposure_idx',
        'Re_wind_exposure_idx',
        'temp_mean', 'temp_diff', 'temp_sd', 'temp_cv', 'pr_mean', 'pr_diff', 'pr_sd', 'pr_cv', 'ws_mean',
        'ws_diff', 'ws_sd', 'ws_cv', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv',
        'Dis_avg', 'Res_time',
        'TN_mean'
    ]
    no_log_cols = ['abs_Pour_lat', 'temp_diff', 'temp_sd', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv']
    for c in compare_cols:
        if c not in no_log_cols:
            post_df[c] = np.log10(post_df[c])


    x_cols = [
        'temp_mean',
        'Depth_avg',
        'Res_time',
        'TN_mean',
        'Lake_area',
        'Shore_dev'
    ]
    ys = ['beta1', 'beta2']
    sig_colors = ['#D98880', '#A2D9CE']
    y_label_dic = {'beta1': r"$\beta_{1}$", 'beta2': r"$\beta_{2}$", 'sigma': r"$\sigma_{i}$"}
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    for i, y in enumerate(ys[:2]):
        rf = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(post_df[x_cols].values, post_df[y].values)
        r2_base = r2_score(post_df[y].values, rf.predict(post_df[x_cols].values))
        rf_importance = pd.Series(rf.feature_importances_, index=x_cols)
        print(y, r2_base)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(post_df[x_cols])
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.Series(shap_abs_mean, index=x_cols)
        shap_importance = shap_importance.sort_values(ascending=False)
        # shap_importance = shap_importance.reset_index()
        most_important_x = shap_importance.index[0]
        print(shap_importance.index[0])
        post_df['shap_value1'] = shap_values[:, x_cols.index(most_important_x)]
        post_df['TN'] = np.power(10, post_df[most_important_x])

        plt.sca(axes[i, 0])
        plt.scatter(post_df['TN'].values, post_df[y].values,
                    alpha=0.6, c='#A4A4A4', edgecolors='k', s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df[y].values, post_df['TN'].values, frac=0.5)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c='#1C2833', ls='-', linewidth=2)
        plt.axhline(0, color='#7F8C8D', ls='--', lw=0.5)
        plt.xscale('log')
        plt.ylabel(y_label_dic[y])
        plt.xlabel('Lake mean TN (µg/L)')
        plt.gca().grid(True, which='both', axis='x')

        plt.sca(axes[i, 1])
        plt.scatter(post_df['TN'].values, post_df['shap_value1'].values,
                    alpha=0.6, c='#A4A4A4', edgecolors='k', s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df['shap_value1'].values, post_df['TN'].values, frac=0.25)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c='#1C2833', ls='-', linewidth=2)
        plt.axhline(0, color='#7F8C8D', ls='--', lw=0.5)

        plt.xscale('log')
        plt.ylabel('SHAP value' + ' for {}'.format(y_label_dic[y]))
        plt.xlabel('Lake mean TN (µg/L)')
        plt.gca().grid(True, which='both', axis='x')

    for ax, label in {
        axes[0, 0]: 'a', axes[0, 1]: 'b',
        axes[1, 0]: 'c', axes[1, 1]: 'd',
    }.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('figs/SI 30 SHAP - TN.png', dpi=300)


def fig31_tp_chla_and_tn_chla():
    fig, axes = plt.subplots(2, 1, figsize=(7, 9))

    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)
    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)
    plt.sca(axes[0])
    plt.scatter(mean_df['TP (µg/L)'], mean_df['Chla (µg/L)'], alpha=0.6, c='#A4A4A4', edgecolors='k', s=30, linewidths=.5)
    lowess_result = sm.nonparametric.lowess(mean_df['Chla (µg/L)'], mean_df['TP (µg/L)'], frac=0.2)
    plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Lake mean Chla (µg/L)')
    plt.xlabel('Lake mean TP (µg/L)')
    slope, intercept, r_value, p_value, std_err = linregress(np.log10(mean_df['TP (µg/L)']), np.log10(mean_df['Chla (µg/L)']))
    plt.title("TP - Chla relationships (n = {}, $r$ = {:.2f})".format(n_lakes, r_value), fontsize=10, fontweight='bold')

    data_df, meta_df = get_lake_data(wq_cols=['Chla (µg/L)', 'TN (µg/L)'], min_wq=1, max_wq=6000, easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])
    n_lakes = data_df['lake_idx'].nunique()
    print('used lakes', n_lakes)
    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)
    plt.sca(axes[1])
    plt.scatter(mean_df['TN (µg/L)'], mean_df['Chla (µg/L)'], alpha=0.6, c='#A4A4A4', edgecolors='k', s=30, linewidths=.5)
    lowess_result = sm.nonparametric.lowess(mean_df['Chla (µg/L)'], mean_df['TN (µg/L)'], frac=0.2)
    plt.plot(lowess_result[:, 0], lowess_result[:, 1], c=line_c, ls='-', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Lake mean Chla (µg/L)')
    plt.xlabel('Lake mean TN (µg/L)')
    slope, intercept, r_value, p_value, std_err = linregress(np.log10(mean_df['TN (µg/L)']), np.log10(mean_df['Chla (µg/L)']))
    plt.title("TN - Chla relationships (n = {}, $r$ = {:.2f})".format(n_lakes, r_value), fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figs/SI 31 TN or TP with Chla.png', dpi=300)


if __name__ == '__main__':
    fig2_global_wq_map_responses_and_n2p_ratio()
    fig3_lake_wq_record_number()
    fig34_lake_tp_chl_box()
    fig4_var_pair_plot()
    fig5_model_var_corr()
    fig6_model_var_sd()
    fig7to9_model_train()
    fig9_1_lake_paras_corr()
    fig10_model_fitness()
    fig11_paras_dist_class()
    fig12_class_diff_box()
    fig13_tree_class()
    fig1314_class_panel_and_high_nutrient_chl()
    fig14_relations()
    fig15to17_shap_explain()
    fig_18_tp_shap_compare()
    fig15to17_shap_explain_for_del_tn()
    fig_18_tp_shap_compare_for_del_tn()
    fig_1920_modeled_P_star_adjust_along_TP()
    fig_20to22_modeled_chl_under_tp_and_t_smooth()
    fig_23_different_min_records()
    fig_24_different_min_months()
    fig_25to27_compare_with_interaction_term()
    fig28_depth_slice()
    fig29to30_nitrogen_relation()
    fig31_tp_chla_and_tn_chla()