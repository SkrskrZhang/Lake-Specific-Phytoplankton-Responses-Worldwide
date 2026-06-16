import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as mcolors
from cycler import cycler


equilibrium_model_paras_dic = {
    'k' : 40,
    'u0' : 0.8,
    'm0' : 0.1,
    'mb' : 0.1,
    'a' : 0.06,
    'b' : 0.09,
    'eta' : 0.02,
    'cpt' : 0.05,
    'cp0' : 0.80,
}



# scatter color and reg lines
scat_c, line_c, fill_c, mark_c, ref_c = '#A4A4A4', '#1C2833', '#3c5e45', '#b03a2e', '#7F8C8D'

class_names = ['NN', 'NP', 'PN', 'PP']
class_c_dic = {
    'NN': '#F2F4F4',
    'NP': '#A2D9CE',
    'PN': '#D98880',
    'PP': '#B03A2E',
}


my_colors = [
    '#54806a',
    '#6a8caf',
    '#c4a468',
    '#a65e5e',
    '#8b7d6b',
    '#5f4b8b',
    '#d58a60',
    '#4b5d67',
    '#95adbe'
]
plt.rcParams['axes.prop_cycle'] = cycler(color=my_colors)

plt.rcParams.update({
    'font.size': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,

    'axes.linewidth': 0.5,
    'axes.labelsize': 9,
    'axes.titlesize': 10,

    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',

    'grid.linewidth': 0.4,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'axes.grid': True,

    "lines.linewidth": 1,


    'figure.figsize': (8, 4),
    # 'figure.autolayout': True,
    'figure.dpi': 100,

    'savefig.dpi': 300,
    # 'savefig.bbox': 'tight',
    # 'savefig.pad_inches': 0.05,
    'savefig.facecolor': 'white',
    'savefig.transparent': False,
})


def p_to_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return None


def get_gradient_color_dict(input_list, cmap=None, cmap_name=None, color_start=None, color_end=None):
    if cmap:
        colormap = cmap
    elif cmap_name:
        colormap = cm.get_cmap(cmap_name)
    else:
        colormap = mcolors.LinearSegmentedColormap.from_list("custom_grad", [color_start, color_end])

    if len(input_list) > 1:
        colors = [mcolors.to_hex(colormap(i)) for i in np.linspace(0, 1, len(input_list))]
        print(colors)
    else:
        colors = [mcolors.to_hex((colormap(0.5)))]
    return dict(zip(input_list, colors))


def adjust_to_darker(color, amount=0.65):
    import colorsys
    safe_amount = min(1.0, amount)

    rgb = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)

    new_l = l * safe_amount

    new_rgb = colorsys.hls_to_rgb(h, new_l, s)
    return mcolors.to_hex(new_rgb)


morandi_green = mcolors.LinearSegmentedColormap.from_list(
        "morandi_green",
        ["#dfe6de",
        "#b8cbb8",
        "#8faa8f",
        "#5f7f65",
        "#3e5f47"
                ]
    )
morandi_warm = mcolors.LinearSegmentedColormap.from_list(
    "morandi_warm",
    [
        "#f2efe8",
        "#e6d8c3",
        "#d6bfa3",
        "#c49a7a",
        "#a8745b",
        "#7f4f3f"
    ]
)