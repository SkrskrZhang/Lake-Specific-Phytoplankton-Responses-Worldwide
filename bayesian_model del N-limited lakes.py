import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import Normal
from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
import random
random.seed(911)

from merge_data import *
from config import *
from uilts import *


def model1():
    data_df, meta_df = get_lake_data(easy_get=True)
    data_df = data_df[data_df['Unique Lake'].isin(meta_df[(meta_df['n_samples'] >= 30) & (meta_df['n_month'] >= 6)].index)]


    # TP and Chla in P-limit lakes
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
    data_df['log_Chla'] = np.log(data_df['Chla (µg/L)'])

    lake_means = data_df.groupby('lake_idx')[['log_TP']].mean()
    lake_means = lake_means.rename(columns={'log_TP': 'mean_log_TP'})
    data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    data_df['log_TP_centered'] = data_df['log_TP'] - data_df['mean_log_TP']

    data_df['temp_centered'] = data_df[temp_col] - 10
    # lake_means = data_df.groupby('lake_idx')[[temp_col]].mean()
    # lake_means = lake_means.rename(columns={temp_col: 'mean_{}'.format(temp_col)})
    # data_df = data_df.merge(lake_means, left_on='lake_idx', right_index=True, how='left')
    # data_df['temp_centered'] = data_df[temp_col] - data_df['mean_{}'.format(temp_col)]


    log_TP = data_df['log_TP'].values
    log_TP_centered = data_df['log_TP_centered'].values
    temp_centered = data_df['temp_centered'].values


    with pm.Model() as model:
        alpha_0 = pm.Normal('alpha_0', mu=0, sigma=5)
        beta1_0 = pm.Normal('beta1_0', mu=0, sigma=5)
        beta2_0 = pm.Normal('beta2_0', mu=0, sigma=5)
        beta3_0 = pm.Normal('beta3_0', mu=0, sigma=5)

        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
        sigma_beta1 = pm.HalfNormal('sigma_beta1', sigma=1)
        sigma_beta2 = pm.HalfNormal('sigma_beta2', sigma=1)
        sigma_beta3 = pm.HalfNormal('sigma_beta3', sigma=1)
        sigma_sigma = pm.HalfNormal('sigma_sigma', sigma=1)

        alpha = pm.Normal('alpha', mu=alpha_0, sigma=sigma_alpha, shape=n_lakes)
        beta1 = pm.Normal('beta1', mu=beta1_0, sigma=sigma_beta1, shape=n_lakes)
        beta2 = pm.Normal('beta2', mu=beta2_0, sigma=sigma_beta2, shape=n_lakes)
        beta3 = pm.Normal('beta3', mu=beta3_0, sigma=sigma_beta3, shape=n_lakes)
        sigma = pm.HalfNormal('sigma', sigma=sigma_sigma, shape=n_lakes)

        # mu = alpha[lake_idx] + beta1[lake_idx] * log_TP_centered + beta2[lake_idx] * temp_centered + beta3[lake_idx] * log_TP_centered * temp_centered
        mu = alpha[lake_idx] + beta1[lake_idx] * log_TP_centered + beta2[lake_idx] * temp_centered

        chla_obs = pm.LogNormal('chla_obs', mu=mu, sigma=sigma[lake_idx], observed=chla)

        trace = pm.sample(draws=2000, tune=2000, target_accept=0.9, random_seed=911)
        ppc = pm.sample_posterior_predictive(trace)


    az.summary(trace)
    az.plot_trace(trace)
    az.plot_autocorr(trace)
    az.plot_pair(trace)
    az.plot_posterior(trace)
    az.ess(trace)
    az.plot_ess(trace)

    alpha_post = trace.posterior["alpha"].values  # (chain, draw, lake)
    beta1_post = trace.posterior["beta1"].values
    beta2_post = trace.posterior["beta2"].values
    beta3_post = trace.posterior["beta3"].values
    sigma_post = trace.posterior["sigma"].values

    n_chain, n_draw, _ = alpha_post.shape
    alpha_post = alpha_post.reshape(-1, n_lakes)  # (chain * draw, lake)
    beta1_post = beta1_post.reshape(-1, n_lakes)
    beta2_post = beta2_post.reshape(-1, n_lakes)
    beta3_post = beta3_post.reshape(-1, n_lakes)
    sigma_post = sigma_post.reshape(-1, n_lakes)

    pd.DataFrame(alpha_post, index=range(n_chain * n_draw), columns=range(n_lakes)).to_csv('results/samples/alpha_post{}.csv'.format(i_study), index=True)
    pd.DataFrame(beta1_post, index=range(n_chain * n_draw), columns=range(n_lakes)).to_csv('results/samples/beta1_post{}.csv'.format(i_study), index=True)
    pd.DataFrame(beta2_post, index=range(n_chain * n_draw), columns=range(n_lakes)).to_csv('results/samples/beta2_post{}.csv'.format(i_study), index=True)
    pd.DataFrame(beta3_post, index=range(n_chain * n_draw), columns=range(n_lakes)).to_csv('results/samples/beta3_post{}.csv'.format(i_study), index=True)
    pd.DataFrame(sigma_post, index=range(n_chain * n_draw), columns=range(n_lakes)).to_csv('results/samples/sigma_post{}.csv'.format(i_study), index=True)
    pd.DataFrame({'idx': range(n_lakes), 'id': lake_codes}).to_csv('results/samples/post_meta{}.csv'.format(i_study), index=False)
    plt.show()

    alpha_post = pd.read_csv('results/samples/alpha_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta1_post = pd.read_csv('results/samples/beta1_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta2_post = pd.read_csv('results/samples/beta2_post{}.csv'.format(i_study), header=0, index_col=0).values
    beta3_post = pd.read_csv('results/samples/beta3_post{}.csv'.format(i_study), header=0, index_col=0).values
    sigma_post = pd.read_csv('results/samples/sigma_post{}.csv'.format(i_study), header=0, index_col=0).values

    beta1_post_low, beta1_post_median, beta1_post_high = np.percentile(beta1_post, [5, 50, 95], axis=0)
    beta2_post_low, beta2_post_median, beta2_post_high = np.percentile(beta2_post, [5, 50, 95], axis=0)
    beta3_post_low, beta3_post_median, beta3_post_high = np.percentile(beta3_post, [5, 50, 95], axis=0)
    n_beta1_sig_pos, n_beta1_pos, n_beta1_sig_neg = (beta1_post_low > 0).sum(), (beta1_post_median > 0).sum(), (beta1_post_high < 0).sum()
    n_beta2_sig_pos, n_beta2_pos, n_beta2_sig_neg = (beta2_post_low > 0).sum(), (beta2_post_median > 0).sum(), (beta2_post_high < 0).sum()
    n_beta3_sig_pos, n_beta3_pos, n_beta3_sig_neg = (beta3_post_low > 0).sum(), (beta3_post_median > 0).sum(), (beta3_post_high < 0).sum()
    print('beta1 {} pos, {} sig pos, {} sig neg'.format(n_beta1_pos / n_lakes, n_beta1_sig_pos / n_lakes, n_beta1_sig_neg / n_lakes))
    print('beta2 {} pos, {} sig pos, {} sig neg'.format(n_beta2_pos / n_lakes, n_beta2_sig_pos / n_lakes, n_beta2_sig_neg / n_lakes))
    print('beta3 {} pos, {} sig pos, {} sig neg'.format(n_beta3_pos / n_lakes, n_beta3_sig_pos / n_lakes, n_beta3_sig_neg / n_lakes))


if __name__ == '__main__':
    model1()
