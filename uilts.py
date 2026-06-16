import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


def detect_outliers_log_mad(df, group_col, value_col):
    outlier_mask = np.zeros(len(df), dtype=bool)

    for lake, idx in df.groupby(group_col).groups.items():
        x = np.log(df.loc[idx, value_col].values)

        med = np.median(x)
        mad = np.median(np.abs(x - med))

        if mad == 0 or np.isnan(mad):
            continue

        z = 0.6745 * (x - med) / mad

        outlier_mask[idx] = np.abs(z) > 3.5

    return outlier_mask


def calc_std_tn(tp):
    return np.power(10, 0.8073 * np.log10(tp) + 1.5015)


def calc_ln(df):
    tp, tn = df['TP (µg/L)'], df['TN (µg/L)']
    std_tn = calc_std_tn(tp)
    std_n2p = std_tn / tp
    df['Limit_Type'] = np.where(tn / tp > std_n2p, 'P', 'N')
    df['LN (µg/L)'] = np.where(tn / tp > std_n2p, tp, tn / std_n2p)
    df['ln_t'] = np.where(tn / tp > std_n2p, 1, 0)
    return df


def calc_chl(tp_arr, t_arr):
    from config import equilibrium_model_paras_dic
    k = equilibrium_model_paras_dic['k']
    u0 = equilibrium_model_paras_dic['u0']
    m0 = equilibrium_model_paras_dic['m0']
    mb = equilibrium_model_paras_dic['mb']
    a = equilibrium_model_paras_dic['a']
    b = equilibrium_model_paras_dic['b']
    eta = equilibrium_model_paras_dic['eta']
    cpt = equilibrium_model_paras_dic['cpt']
    cp0 = equilibrium_model_paras_dic['cp0']

    p_star = k / ((u0 * np.exp(a * t_arr) / (mb + m0 * np.exp(b * t_arr))) - 1)
    p_star_adjust = p_star * (tp_arr / (tp_arr + p_star))
    sim_chl = 2 + (cp0 * np.exp(cpt * t_arr) * (tp_arr - p_star_adjust)) / (1 + eta * tp_arr ** 1)
    return sim_chl