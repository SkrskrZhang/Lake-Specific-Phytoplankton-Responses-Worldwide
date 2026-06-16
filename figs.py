import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import Normal
from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
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
from scipy.optimize import brentq
import matplotlib.gridspec as gridspec
from scipy.interpolate import UnivariateSpline
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
from matplotlib.patches import Rectangle
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

import random
random.seed(911)
np.random.seed(911)

from merge_data import *
from config import *
from uilts import *


def fig1():
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
    beta3_post = pd.read_csv('results/samples/beta3_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values


    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)
    # beta3_post_low, beta3_post_median, beta3_post_high = np.percentile(beta3_post, [5, 50, 95], axis=0)
    n_beta1_sig_pos, n_beta1_pos, n_beta1_sig_neg = (beta1_post_low > 0).sum(), (beta1_post_median > 0).sum(), (beta1_post_high < 0).sum()
    n_beta2_sig_pos, n_beta2_pos, n_beta2_sig_neg = (beta2_post_low > 0).sum(), (beta2_post_median > 0).sum(), (beta2_post_high < 0).sum()
    print('beta1 {} pos, {} sig pos, {} sig neg'.format(n_beta1_pos / n_lakes, n_beta1_sig_pos / n_lakes, n_beta1_sig_neg / n_lakes))
    print('beta2 {} pos, {} sig pos, {} sig neg'.format(n_beta2_pos / n_lakes, n_beta2_sig_pos / n_lakes, n_beta2_sig_neg / n_lakes))


    post_df = pd.DataFrame(index=lake_codes)
    post_df.index.name = 'Unique Lake'
    post_df['beta1'] = beta1_post_median
    post_df['beta2'] = beta2_post_median
    post_df['sigma'] = np.percentile(sigma_post, 50, axis=0)
    post_df['cv'] = np.percentile(np.sqrt(np.exp(sigma_post) - 1), 50, axis=0)
    post_df['alpha'] = np.percentile(alpha_post, 50, axis=0)

    post_df['beta1_sig_pos'] = beta1_post_low > 0
    post_df['beta2_sig_pos'] = beta2_post_low > 0
    post_df['beta1_sig_neg'] = beta1_post_high < 0
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
        'Lake_area', 'Shore_dev', 'Depth_avg', 'Wshd_lake_ratio', 'Vol_total', 'Mix_depth_ratio', 'Wind_exposure_idx', 'Re_wind_exposure_idx',
        'temp_mean', 'temp_diff', 'temp_sd', 'temp_cv', 'pr_mean', 'pr_diff', 'pr_sd', 'pr_cv', 'ws_mean',
        'ws_diff', 'ws_sd', 'ws_cv', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv',
        'Dis_avg', 'Res_time',
        'TP_mean', 'TP_sd', 'TP_cv',
              ]
    no_log_cols = ['abs_Pour_lat', 'temp_diff', 'temp_sd', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv', 'temp_mean']
    for c in compare_cols:
        if c not in no_log_cols:
            post_df[c] = np.log10(post_df[c])
    post_df.to_csv('post_df - {}.csv'.format(i_study))


    post_df['G1_PP'] = (post_df['beta1_sig_pos'] & post_df['beta2_sig_pos'])
    post_df['G2_PN'] = (post_df['beta1_sig_pos'] & ~post_df['beta2_sig_pos'])
    post_df['G3_NP'] = (~post_df['beta1_sig_pos'] &post_df['beta2_sig_pos'])
    post_df['G4_NN'] = (~post_df['beta1_sig_pos'] &~post_df['beta2_sig_pos'])
    conditions = [post_df['G1_PP'],post_df['G2_PN'],post_df['G3_NP'],post_df['G4_NN']]
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
        'temp_mean',
        'Depth_avg',
        'Res_time',
        'TP_mean',
        'Lake_area',
        'Shore_dev'
    ]
    X = post_df[x_cols]
    y = post_df['Class_num']
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=911)
    clf.fit(X, y)
    print(clf.tree_.children_left)


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 5], width_ratios=[4, 4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])


    ax_cols = ['beta1', 'beta2']
    ax_titles = [r'TP coefficient', r'T coefficient']
    ax_inset_titles = [r"Sig. of $\beta_{1,i}$", r"Sig. of $\beta_{2,i}$"]
    ax_pie_colors = [['#D98880', '#85929E', '#ECF0F1'], ['#A2D9CE', '#85929E', '#ECF0F1']]
    ax_x_labels = [r'Posterior distribution of hyperparameter $\beta_1$', r'Posterior distribution of hyperparameter $\beta_2$']
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
        counts = [len(post_df[post_df['{}_sig_pos'.format(ax_cols[i_a])]>0]), len(post_df[post_df['{}_sig_neg'.format(ax_cols[i_a])]>0])]
        counts.append(n_lakes - counts[0] - counts[1])
        ax_inset.pie(counts, labels=labels, colors=ax_pie_colors[i_a], autopct='%1.0f%%', textprops={'fontsize': 7}, startangle=90)
        ax_inset.set_title(ax_inset_titles[i_a], fontsize=8)
        plt.xlabel('{}\n({})'.format(ax_x_labels[i_a], ax_titles[i_a]))
        if i_a == 0:
            plt.ylabel("Density")


    def get_leaf_boundaries(clf, feature_names, x_feat, y_feat, x_range, y_range):
        tree_ = clf.tree_
        leaf_regions = []

        def recurse(node, x1, x2, y1, y2):
            if tree_.children_left[node] == _tree.TREE_LEAF:
                leaf_regions.append({
                    'box': [x1, x2, y1, y2],
                    'counts': tree_.value[node][0] * tree_.n_node_samples[node],
                    'n': tree_.n_node_samples[node]
                })
                return
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            if feature == x_feat:
                recurse(tree_.children_left[node], x1, threshold, y1, y2)
                recurse(tree_.children_right[node], threshold, x2, y1, y2)
            elif feature == y_feat:
                recurse(tree_.children_left[node], x1, x2, y1, threshold)
                recurse(tree_.children_right[node], x1, x2, threshold, y2)
            else:
                recurse(tree_.children_left[node], x1, x2, y1, y2)
                recurse(tree_.children_right[node], x1, x2, y1, y2)

        recurse(0, x_range[0], x_range[1], y_range[0], y_range[1])
        return leaf_regions

    def merge_specific_regions(regions, indices_to_merge):
        if not indices_to_merge:
            return regions

        to_be_merged = [regions[i] for i in indices_to_merge]

        merged_counts = np.sum([r['counts'] for r in to_be_merged], axis=0)
        merged_n = sum([r['n'] for r in to_be_merged])

        all_x1 = [r['box'][0] for r in to_be_merged]
        all_x2 = [r['box'][1] for r in to_be_merged]
        all_y1 = [r['box'][2] for r in to_be_merged]
        all_y2 = [r['box'][3] for r in to_be_merged]

        merged_box = [min(all_x1), max(all_x2), min(all_y1), max(all_y2)]

        merged_region = {
            'box': merged_box,
            'counts': merged_counts,
            'n': merged_n
        }

        new_regions = regions.copy()
        for i in sorted(indices_to_merge, reverse=True):
            new_regions.pop(i)
        new_regions.append(merged_region)

        return new_regions

    plt.sca(ax3)
    ax = plt.gca()
    colors = ['#F2F4F4', '#A2D9CE', '#D98880', '#B03A2E']
    name_dic = {'TP_mean': 'TP', 'temp_mean': 'T'}
    unit_dic = {'TP_mean': ' µg/L', 'temp_mean': '°C'}

    x_feature = 'TP_mean'
    y_feature = 'temp_mean'
    x_min, x_max = 0.5, 2.3
    y_min, y_max = 0, 27
    regions = get_leaf_boundaries(clf, x_cols, x_feature, y_feature, [x_min, x_max], [y_min, y_max])

    regions = merge_specific_regions(regions, [0, 1])
    regions = merge_specific_regions(regions, [0, 1])


    for class_name in class_names:
        sub_df = post_df[post_df['Class'] == class_name]
        x, y = sub_df['TP_mean'].values, sub_df['temp_mean'].values
        plt.scatter(x, y, s=15, alpha=0.95, color=class_c_dic[class_name], linewidth=0.2, edgecolors=line_c, label=class_name, zorder=-1)


    a_ls = [3,4,5,6,1,2]
    for i_a, reg in enumerate(regions):
        x1, x2, y1, y2 = reg['box']
        print(reg)

        split_c, alpha, lw = scat_c, 0.6, 0.5
        plt.plot([1.0476744771003723, 1.0476744771003723], [y_min, y_max], color=split_c, linestyle='-', lw=lw, alpha=alpha)
        plt.plot([0.5, 1.0476744771003723], [7.336688280105591, 7.336688280105591], color=split_c, linestyle='-', lw=lw, alpha=alpha)
        plt.plot([1.4969202876091003, 1.4969202876091003], [0, 16.1571364402771], color=split_c, linestyle='-', lw=lw, alpha=alpha)
        plt.plot([1.0476744771003723, x_max], [16.1571364402771, 16.1571364402771], color=split_c, linestyle='-', lw=lw, alpha=alpha)
        plt.plot([1.9525858759880066, 1.9525858759880066], [16.1571364402771, y_max], color=split_c, linestyle='-', lw=lw, alpha=alpha)

        text_c, fs = scat_c, 8
        ax.text(1.0476744771003723, 0, '11', ha='center', va='bottom', fontsize=fs, color=text_c)
        ax.text(1.4969202876091003, 0, '31', ha='center', va='bottom', fontsize=fs, color=text_c)
        ax.text(0.5, 7.336688280105591, '7', ha='left', va='center', fontsize=fs, color=text_c)
        ax.text(1.9525858759880066, y_max, '90', ha='center', va='bottom', fontsize=fs, color=text_c)
        ax.text(x_max, 16.1571364402771, '16', ha='left', va='center', fontsize=fs, color=text_c)

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        size = np.interp(reg['n'], [15, 270], [4, 10])

        ax_inset = ax.inset_axes([cx - size / 2, cy - size / 2, size, size], transform=ax.transData)
        ax_inset.pie(reg['counts'], colors=colors, startangle=90, autopct=lambda p: f'{p:.0f}%' if p > 40 else '', textprops={'fontsize': 7})

        ax.text(cx, cy - size / 2 - 0.02, "A{}: n={}".format(a_ls[i_a], reg['n']), ha='center', va='top', fontsize=8)

    ax.set_xlabel('Lake mean TP (µg/L)')
    ax.set_ylabel('Lake mean T (°C)')
    plt.xlim(0.5, 2.4)
    plt.ylim(0, 29)
    plt.xticks([1, 2], ['10$^{1}$', '10$^{2}$'])
    ax.xaxis.set_minor_locator(ticker.FixedLocator([np.log10(mx) for mx in [5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90, 200]]))
    plt.yticks([0, 5, 10, 15, 20, 25])
    plt.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    rect = Rectangle(
        (np.log10(5), 21.5),np.log10(10)-np.log10(5), 7.5,
        facecolor='grey',
        edgecolor='grey',
        linewidth=0,
        alpha=0.1
    )
    ax.add_patch(rect)

    ax_leg = ax.inset_axes([0.15, 0.82, 0.1, 0.16])
    matrix_colors = [
        [colors[0], colors[2]],
        [colors[1], colors[3]]
    ]
    matrix_class = [
        ['NN', 'PN'],
        ['NP', 'PP']
    ]
    matrix_values = [
        [lake_counts['NN'], lake_counts['PN']],
        [lake_counts['NP'], lake_counts['PP']]
    ]
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j, i), 1, 1, facecolor=matrix_colors[i][j], edgecolor='white', lw=1)
            ax_leg.add_patch(rect)
            ax_leg.text(j + 0.5, i + 0.5,
                        '{}\n{:.0f}%'.format(matrix_class[i][j], (matrix_values[i][j] / n_lakes) * 100),
                        ha='center', va='center', fontsize=7)
    ax_leg.set_xlim(0, 2)
    ax_leg.set_ylim(0, 2)
    ax_leg.set_xticks([0.5, 1.5])
    ax_leg.set_yticks([0.85, 1.6])
    ax_leg.set_xticklabels(['n.s./—', '+'], fontsize=6)
    ax_leg.set_yticklabels(['n.s./—', '+'], fontsize=6, rotation=90)
    ax_leg.set_xlabel(r'$\beta_{1,i}$', fontsize=7, labelpad=1)
    ax_leg.set_ylabel(r'$\beta_{2,i}$', fontsize=7, labelpad=1)
    ax_leg.spines['top'].set_visible(False)
    ax_leg.spines['right'].set_visible(False)
    ax_leg.spines['bottom'].set_visible(False)
    ax_leg.spines['left'].set_visible(False)
    ax_leg.tick_params(size=0)
    ax_leg.grid(False)
    ax_leg.set_facecolor('grey')
    ax_leg.patch.set_alpha(1.0)

    for ax, label in {ax1: 'a', ax2: 'b', ax3: 'c'}.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.savefig('figs/MS 1.png', dpi=300)
    plt.show()


def fig2():
    data_df, meta_df = get_lake_data(easy_get=True)

    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]

    # without interaction term
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
    no_log_cols = ['abs_Pour_lat', 'temp_diff', 'temp_sd', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv']
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

    ys = ['beta1', 'beta2', 'sigma']
    sig_colors = ['#D98880', '#A2D9CE']
    y_label_dic = {'beta1': r"$\beta_{1} ± SD$", 'beta2': r"$\beta_{2} ± SD$", 'sigma': r"$\sigma_{i}$"}
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
        print(most_important_x)
        post_df['shap_value1'] = shap_values[:, x_cols.index(most_important_x)]
        post_df['TP'] = np.power(10, post_df[most_important_x])

        plt.sca(axes[i, 0])
        plt.errorbar(post_df['TP'].values, post_df[y].values, yerr=post_df[y + "_sd"].values,
                     fmt='none', ecolor='gray', elinewidth=0.8, alpha=0.5, capsize=0, zorder=1)
        plt.scatter(post_df['TP'].values, post_df[y].values,
                    alpha=0.6, c='#A4A4A4', edgecolors='k', s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df[y].values, post_df['TP'].values, frac=0.4)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c='#1C2833', ls='-', linewidth=2)
        plt.axhline(0, color='#7F8C8D', ls='--', lw=0.5)
        plt.xscale('log')
        plt.ylabel(y_label_dic[y])
        plt.xlabel('Lake mean TP (µg/L)')
        plt.gca().grid(True, which='both', axis='x')


        plt.sca(axes[i, 1])
        plt.scatter(post_df['TP'].values, post_df['shap_value1'].values,
                        alpha=0.6, c='#A4A4A4', edgecolors='k', s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df['shap_value1'].values, post_df['TP'].values, frac=0.25)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c='#1C2833', ls='-', linewidth=2)
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
            plt.text((10**tau2_median) * 1.1, 0.3, 'Saturation\nthreshold: {:.0f}'.format(10**tau2_median), transform=plt.gca().get_xaxis_transform(), fontsize=9, va='center', ha='left', color=mark_c, multialignment='left')

        plt.xscale('log')
        plt.ylabel('SHAP value' + ' for {}'.format(y_label_dic[y].replace(' ± SD', '')))
        plt.xlabel('Lake mean TP (µg/L)')
        plt.gca().grid(True, which='both', axis='x')

    for ax, label in {
        axes[0, 0]: 'a', axes[0, 1]: 'b',
        axes[1, 0]: 'c', axes[1, 1]: 'd',
    }.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('figs/MS 2.png', dpi=300)


def fig3():
    data_df, meta_df = get_lake_data(easy_get=True)

    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]
    ini_data_df = copy.deepcopy(data_df)

    # without n-limited lakes
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
    no_log_cols = ['abs_Pour_lat', 'temp_diff', 'temp_sd', 'sr_mean', 'sr_diff', 'sr_sd', 'sr_cv']
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

    ys = ['beta1', 'beta2', 'sigma']
    y_label_dic = {'beta1': r"$\beta_{1} ± SD$", 'beta2': r"$\beta_{2} ± SD$", 'sigma': r"$\sigma_{i}$"}
    fig, axes = plt.subplots(1, 2, figsize=(8, 7.5/2))
    axes = axes.reshape(1, 2)
    for i, y in enumerate(ys[:1]):
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
        print(most_important_x)
        post_df['shap_value1'] = shap_values[:, x_cols.index(most_important_x)]
        post_df['TP'] = np.power(10, post_df[most_important_x])

        plt.sca(axes[i, 0])
        plt.errorbar(post_df['TP'].values, post_df[y].values, yerr=post_df[y + "_sd"].values,
                     fmt='none', ecolor='gray', elinewidth=0.8, alpha=0.5, capsize=0, zorder=1)
        plt.scatter(post_df['TP'].values, post_df[y].values,
                    alpha=0.6, c='#A4A4A4', edgecolors='k', s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df[y].values, post_df['TP'].values, frac=0.35)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c='#1C2833', ls='-', linewidth=2)
        plt.axhline(0, color='#7F8C8D', ls='--', lw=0.5)
        plt.xscale('log')
        plt.ylabel(y_label_dic[y])
        plt.xlabel('Lake mean TP (µg/L)')
        plt.gca().grid(True, which='both', axis='x')


        plt.sca(axes[i, 1])
        plt.scatter(post_df['TP'].values, post_df['shap_value1'].values,
                        alpha=0.6, c='#A4A4A4', edgecolors='k', s=30, linewidths=.5)
        lowess_result = sm.nonparametric.lowess(post_df['shap_value1'].values, post_df['TP'].values, frac=0.25)
        plt.plot(lowess_result[:, 0], lowess_result[:, 1], c='#1C2833', ls='-', linewidth=2)
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
            plt.text((10**tau2_median) * 1.1, 0.3, 'Saturation\nthreshold: {:.0f}'.format(10**tau2_median), transform=plt.gca().get_xaxis_transform(), fontsize=9, va='center', ha='left', color=mark_c, multialignment='left')

        plt.xscale('log')
        plt.ylabel('SHAP value' + ' for {}'.format(y_label_dic[y].replace(' ± SD', '')))
        plt.xlabel('Lake mean TP (µg/L)')
        plt.gca().grid(True, which='both', axis='x')

    for ax, label in {
        axes[0, 0]: 'a', axes[0, 1]: 'b',
    }.items():
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('figs/MS 3.png', dpi=300)


def fig4():
    data_df, meta_df = get_lake_data(easy_get=True)

    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]

    data_df['lake_idx'], lake_codes = pd.factorize(data_df['Unique Lake'])

    mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)





    fig, axes = plt.subplots(2, 1, figsize=(8, 9), height_ratios=[3.5, 5.5])
    ax_model_heatmap = axes[0]
    ax_global_p2c = axes[1]


    plt.sca(ax_model_heatmap)
    x = np.linspace(3, 200, 100)
    y = np.linspace(0, 30, 80)
    X, Y = np.meshgrid(x, y, indexing='ij')
    grid = np.stack((X, Y), axis=-1).reshape(-1, 2)  # n_x, n_y, 2 -> n_x * n_y, 2

    tp_arr, t_arr = grid[:, 0], grid[:, 1]
    sim_chl = calc_chl(tp_arr, t_arr)
    Z = sim_chl.reshape(len(x), len(y))

    heatmap = plt.pcolormesh(X, Y, Z, cmap=morandi_green, norm=LogNorm(), shading='gouraud')
    cbar = plt.colorbar(heatmap, pad=0.03, fraction=0.05)
    plt.xscale('log')
    plt.xlim(3, 200)
    plt.ylim(0, 30)
    cbar.ax.set_yscale('log')
    cbar.set_label('Modeled equilibrium Chl$a$ (µg/L)')
    plt.xlabel('TP (µg/L)', labelpad=0)
    plt.ylabel('T (°C)')


    plt.sca(ax_global_p2c)
    plt.scatter(mean_df['TP (µg/L)'], mean_df['Chla (µg/L)'], alpha=0.7, c='#A4A4A4', edgecolors='k', s=30, linewidths=.5, zorder=99, label='577 studied lakes')
    lowess_result = sm.nonparametric.lowess(mean_df['Chla (µg/L)'], mean_df['TP (µg/L)'], frac=0.2)
    plt.plot(lowess_result[:, 0], lowess_result[:, 1], c='#1C2833', ls='-', linewidth=2, zorder=100, label='LOWESS')


    t_slice = [0, 10, 20, 30]
    tp_arr = np.linspace(3, 200, 1000)
    sampled_temp_colors = get_gradient_color_dict(t_slice, cmap=morandi_warm)
    for t in t_slice:
        sim_chl = calc_chl(tp_arr, t)
        plt.plot(tp_arr, sim_chl, c=sampled_temp_colors[t], ls='-', linewidth=2, label='Modeled equilibrium Chl$a$ at T = {} °C'.format(t), zorder=101)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Lake mean TP (µg/L)', labelpad=0)
    plt.ylabel('Lake mean Chl$a$ (µg/L)')


    plt.legend(frameon=False, fontsize=8, loc='upper left')


    for ax, label in {ax_global_p2c: 'b', ax_model_heatmap: 'a'}.items():
        ax.text(0.025, 0.95 if label=='a' else 0.75, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left', zorder=999)


    plt.tight_layout()
    plt.subplots_adjust(hspace=.2)
    plt.savefig('figs/MS 4.png', dpi=300)


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
