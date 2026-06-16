import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import Normal
from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.tri as tri
import chardet
from statsmodels.genmod.families import Family, Gaussian, Poisson, Binomial, Gamma, NegativeBinomial
from statsmodels.genmod.families.links import identity, log, logit, inverse_power
from scipy.stats import kruskal, mannwhitneyu, gamma
from itertools import combinations
import matplotlib as mpl
import warnings
import seaborn as sns
from sympy.abc import alpha
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')


def filter_summer_data(data_df, meta_df, c_date='sampledate', c_lat='Pour_lat', t_lat=5):
    summer_months = [6, 7, 8]
    summer_months_s = [12, 1, 2]

    n_idx = meta_df[meta_df[c_lat] > t_lat].index
    s_idx = meta_df[meta_df[c_lat] < -t_lat].index
    o_idx = meta_df[(meta_df[c_lat] < t_lat) & (meta_df[c_lat] > -t_lat)].index

    data_df[c_date] = pd.to_datetime(data_df[c_date])
    data_df['year'] = data_df[c_date].dt.year
    data_df['month'] = data_df[c_date].dt.month

    o_data_df = data_df[data_df['Unique Lake'].isin(o_idx)]
    n_data_df = data_df[data_df['Unique Lake'].isin(n_idx)]
    s_data_df = data_df[data_df['Unique Lake'].isin(s_idx)]
    n_summer_data_df = n_data_df[n_data_df['month'].isin(summer_months)]
    s_summer_data_df = s_data_df[s_data_df['month'].isin(summer_months_s)]
    data_df = pd.concat([o_data_df, n_summer_data_df, s_summer_data_df], axis=0)
    return data_df


def load_temp(data_df, meta_df):
    hylak_temp_df =  pd.read_csv('data/Hylak_monthly_temperature_2m.csv', header=0)[['Hylak_id', 'month', 'mean']]
    hylak_temp_df = hylak_temp_df.dropna()
    hylak_temp_df['mean'] -= 273.15

    c_date = 'sampledate'
    data_df[c_date] = pd.to_datetime(data_df[c_date])
    data_df['year'] = data_df[c_date].dt.year
    data_df['month'] = data_df[c_date].dt.month

    meta_rsi_df = meta_df.reset_index()
    data_df = data_df.merge(meta_rsi_df[['Unique Lake', 'Hylak_id']], on='Unique Lake', how='left')

    data_df = data_df.merge(hylak_temp_df, on=['Hylak_id', 'month'], how='left')

    data_df = data_df.rename(columns={'mean': 'temp'})

    n_nan = data_df['temp'].isna().sum()
    print('no temp values', n_nan, n_nan / len(data_df))

    return data_df, hylak_temp_df


def load_temp_from_awq(data_df, meta_df):
    for name in ['lake_mix_layer_temperature', 'temperature_2m']:
        awqlak_temp_df =  pd.read_csv('data/awqlak_monthly_{}_1980.csv'.format(name), header=0)[['Unique Lak', 'year', 'month', 'mean']].rename(columns={'Unique Lak': 'Unique Lake'})
        print('awqlak records', len(awqlak_temp_df))
        awqlak_temp_df = awqlak_temp_df.dropna()
        awqlak_temp_df['mean'] -= 273.15
        print('awqlak records', len(awqlak_temp_df))

        data_df = data_df.merge(awqlak_temp_df, on=['Unique Lake', 'year', 'month'], how='left').rename(columns={'mean': 'awq_{}'.format(name)})
        clim_80_00 = (
            awqlak_temp_df
            .query('1980 <= year <= 2000')
            .groupby(['Unique Lake', 'month'])['mean']
            .mean()
            .reset_index()
            .rename(columns={'mean': 'clim_mean'})
        )
        data_df = data_df.merge(
            clim_80_00,
            on=['Unique Lake', 'month'],
            how='left'
        )
        data_df['awq_{}'.format(name)] = data_df['awq_{}'.format(name)].fillna(data_df['clim_mean'])
        data_df = data_df.drop(columns='clim_mean')
        n_nan = data_df['awq_{}'.format(name)].isna().sum()
        print('awq no {} values'.format(name), n_nan, n_nan / len(data_df))

    return data_df


def load_mere_vars():
    hylak_pr_df = pd.read_csv('data/Hylak_monthly_total_precipitation_sum.csv', header=0)[['Hylak_id', 'month', 'mean']]
    hylak_pr_df['mean'] *= 1e4
    hylak_pr_df = hylak_pr_df.dropna()

    hylak_ws_u_df = pd.read_csv('data/Hylak_monthly_u_component_of_wind_10m.csv', header=0)[['Hylak_id', 'month', 'mean']]
    hylak_ws_v_df = pd.read_csv('data/Hylak_monthly_v_component_of_wind_10m.csv', header=0)[['Hylak_id', 'month', 'mean']]
    hylak_ws_df = copy.deepcopy(hylak_ws_u_df)
    hylak_ws_df['mean'] = np.sqrt(hylak_ws_u_df['mean'] ** 2 + hylak_ws_v_df['mean'] ** 2)
    hylak_ws_df = hylak_ws_df.dropna()

    hylak_sr_df = pd.read_csv('data/Hylak_monthly_surface_solar_radiation_downwards_sum.csv', header=0)[['Hylak_id', 'month', 'mean']]
    hylak_sr_df['mean'] *= 1e-6
    hylak_sr_df = hylak_sr_df.dropna()
    return hylak_pr_df, hylak_ws_df, hylak_sr_df


def get_lake_data(wq_cols=['Chla (µg/L)', 'TP (µg/L)'], min_wq=1, max_wq=200, easy_get=False, if_show_fig=False, max_wq_ln=200):
    """
    water quality df: row with samples
    lake meta df: row with unique lake
    """
    wq_str = ''
    for wq in wq_cols:
        wq_str = wq_str + wq[:2]

    if easy_get:
        if os.path.exists('data/meta {} {}-{}.csv'.format(wq_str, min_wq, max_wq)) and os.path.exists('data/data {} {}-{}.csv'.format(wq_str, min_wq, max_wq)):
            lake_meta_df = pd.read_csv('data/meta {} {}-{}.csv'.format(wq_str, min_wq, max_wq), header=0, index_col='Unique Lake')
            data_df = pd.read_csv('data/data {} {}-{}.csv'.format(wq_str, min_wq, max_wq), header=0, index_col='Sample ID')
            return data_df, lake_meta_df


    awq_data_df = pd.read_csv('data/Lake_Metadata.csv', header=0, encoding='latin1',
                              dtype={7: "string", 9: "string", 11: "string", 15: "string", 17: "string", 22: "string", 24: "string"})
    awq_data_df['sampledate'] = pd.to_datetime(awq_data_df['sampledate'])
    data_df = awq_data_df
    print('ini wq data records', len(data_df))

    cltp_data_df = data_df.dropna(subset=['Chla (µg/L)', 'TP (µg/L)'])
    cltp_data_mean_df = cltp_data_df.groupby('Unique Lake').mean(numeric_only=True)
    print('ini cl tp data records', len(cltp_data_df), len(cltp_data_mean_df))

    cltn_data_df = data_df.dropna(subset=['Chla (µg/L)', 'TN (µg/L)'])
    cltn_data_mean_df = cltn_data_df.groupby('Unique Lake').mean(numeric_only=True)
    print('ini cl tn data records', len(cltn_data_df), len(cltn_data_mean_df))

    cltnp_data_df = data_df.dropna(subset=['Chla (µg/L)', 'TN (µg/L)', 'TP (µg/L)'])
    cltnp_data_mean_df = cltnp_data_df.groupby('Unique Lake').mean(numeric_only=True)
    print('ini cl tnp data records', len(cltnp_data_df), len(cltnp_data_mean_df))


    data_df = data_df.dropna(subset=['sampledate']+wq_cols)
    for wq in wq_cols:
        data_df = data_df[(data_df[wq] > min_wq)]

    if 'TN (µg/L)' in wq_cols and 'TP (µg/L)' in wq_cols:
        from uilts import calc_ln
        data_df = calc_ln(data_df)

    wq_mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)
    if 'TN (µg/L)' in wq_cols and 'TP (µg/L)' in wq_cols:
        data_df = data_df[data_df['Unique Lake'].isin(wq_mean_df[wq_mean_df['TN (µg/L)'] < max_wq].index)]
        data_df = data_df[data_df['Unique Lake'].isin(wq_mean_df[wq_mean_df['TP (µg/L)'] < max_wq_ln].index)]
        data_df = data_df[data_df['Unique Lake'].isin(wq_mean_df[wq_mean_df['LN (µg/L)'] < max_wq_ln].index)]
        data_df = data_df[data_df['Unique Lake'].isin(wq_mean_df[wq_mean_df['Chla (µg/L)'] < max_wq_ln].index)]
    else:
        for wq in wq_cols:
            data_df = data_df[data_df['Unique Lake'].isin(wq_mean_df[wq_mean_df[wq] < max_wq].index)]

    print('both date/chl/tp data records', len(data_df))


    lake_meta_df = pd.read_csv('data/Lake_Altitude.csv', header=0, encoding='latin1').loc[:, ['Unique Lake', 'Latitude', 'Longitude', 'Altitude (m)']]
    lake_meta_df = lake_meta_df.groupby('Unique Lake').mean()
    lake_meta_df = lake_meta_df.dropna(subset=['Latitude', 'Longitude'])
    print('lake meta records', len(lake_meta_df))


    wq_mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)
    idx = wq_mean_df.index.intersection(lake_meta_df.index)
    print('diff in wq and meta', len(wq_mean_df), len(idx))


    hylak_df = pd.read_csv('data/hydro_lake_data.csv', header=0, index_col=0)
    print('hylak records', len(hylak_df), hylak_df.columns)


    coords_all = np.column_stack((hylak_df['Pour_long'].values, hylak_df['Pour_lat'].values))
    coords_local = np.column_stack((lake_meta_df['Longitude'].values, lake_meta_df['Latitude'].values))
    tree = cKDTree(coords_all)
    dist, idx = tree.query(coords_local, k=1)
    lake_meta_df['Hylak_id'] = hylak_df.index.values[idx]


    lake_meta_df = lake_meta_df.join(hylak_df, on='Hylak_id')
    print('joined lake meta df', lake_meta_df, lake_meta_df.columns)


    data_df, hylak_temp_df = load_temp(data_df, lake_meta_df)
    print(data_df.head(), lake_meta_df.head())
    data_df = data_df.dropna(subset=wq_cols+['temp'])


    data_df = load_temp_from_awq(data_df, lake_meta_df)
    print(data_df.head(), data_df.columns)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (x, y) in enumerate([['temp', 'awq_temperature_2m'], ['temp', 'awq_lake_mix_layer_temperature'], ['awq_temperature_2m', 'awq_lake_mix_layer_temperature']]):
        plt.sca(axes[i])
        plt.scatter(data_df[x], data_df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.plot([0, 40], [0, 40], ls='--', c='r')
        plt.grid()
    plt.tight_layout()
    data_df = data_df.dropna(subset=wq_cols+['awq_temperature_2m', 'awq_lake_mix_layer_temperature'])


    hylak_pr_df, hylak_ws_df, hylak_sr_df = load_mere_vars()


    for name, df in {'temp': hylak_temp_df, 'pr': hylak_pr_df, 'ws': hylak_ws_df, 'sr': hylak_sr_df}.items():
        hylak_temp_mean_df = df.groupby('Hylak_id').mean().rename(columns={'mean': '{}_mean'.format(name)})
        lake_meta_df = lake_meta_df.join(hylak_temp_mean_df[['{}_mean'.format(name)]], on='Hylak_id')
        hylak_temp_diff_df = (df.groupby('Hylak_id').max() - df.groupby('Hylak_id').min()).rename(columns={'mean': '{}_diff'.format(name)})
        lake_meta_df = lake_meta_df.join(hylak_temp_diff_df[['{}_diff'.format(name)]], on='Hylak_id')
        hylak_temp_sd_df = df.groupby('Hylak_id').std().rename(columns={'mean': '{}_sd'.format(name)})
        lake_meta_df = lake_meta_df.join(hylak_temp_sd_df[['{}_sd'.format(name)]], on='Hylak_id')
        hylak_temp_cv_df = (df.groupby('Hylak_id').std() / df.groupby('Hylak_id').mean()).rename(columns={'mean': '{}_cv'.format(name)})
        lake_meta_df = lake_meta_df.join(hylak_temp_cv_df[['{}_cv'.format(name)]], on='Hylak_id')

    for wq in wq_cols:
        wq_mean_df = data_df.groupby('Unique Lake').mean(numeric_only=True)[[wq]]
        wq_sd_df = data_df.groupby('Unique Lake').std(numeric_only=True)[[wq]]
        wq_cv_df = (wq_sd_df / wq_mean_df)
        wq_stat_df =pd.concat([wq_mean_df.rename(columns={wq: '{}_mean'.format(wq[:2])}), wq_sd_df.rename(columns={wq: '{}_sd'.format(wq[:2])}), wq_cv_df.rename(columns={wq: '{}_cv'.format(wq[:2])})], axis=1)
        lake_meta_df = lake_meta_df.join(wq_stat_df, on='Unique Lake')
    print(lake_meta_df.columns)


    lake_counts = data_df.groupby('Unique Lake').size()
    idx = lake_counts.index.intersection(lake_meta_df.index)
    lake_meta_df.loc[idx, 'n_samples'] = lake_counts.loc[idx].values
    print(lake_meta_df.head(), lake_meta_df.columns)


    n_record_month_df = data_df.groupby('Unique Lake')[['year', 'month']].nunique().rename(columns={'year': 'n_year', 'month': 'n_month'})
    lake_meta_df = lake_meta_df.join(n_record_month_df, on='Unique Lake')
    month_diff_df = (data_df.groupby('Unique Lake')[['month']].max() - data_df.groupby('Unique Lake')[['month']].min()).rename(columns={'month': 'month_diff'})
    lake_meta_df = lake_meta_df.join(month_diff_df, on='Unique Lake')
    print(lake_meta_df.head(), lake_meta_df.columns)


    lake_meta_df['mixing_depth'] = (np.power(10, 0.185 * np.log10(lake_meta_df['Lake_area']) + 0.842) - 2.37) / 1.05


    lake_meta_df.to_csv('data/meta {} {}-{}.csv'.format(wq_str, min_wq, max_wq), index_label='Unique Lake')
    data_df.to_csv('data/data {} {}-{}.csv'.format(wq_str, min_wq, max_wq), index_label='Sample ID')
    print('used lakes', len(data_df.groupby('Unique Lake').mean(numeric_only=True)))


    if if_show_fig:
        plt.show()


    return data_df, lake_meta_df


if __name__ == '__main__':
    get_lake_data(wq_cols=['Chla (µg/L)', 'TP (µg/L)'], min_wq=1, max_wq=200, easy_get=False)
    # get_lake_data(wq_cols=['Chla (µg/L)', 'TP (µg/L)', 'TN (µg/L)'], min_wq=1, max_wq=6000, max_wq_ln=200)
    # get_lake_data(wq_cols=['Chla (µg/L)', 'TN (µg/L)'], min_wq=1, max_wq=6000)
