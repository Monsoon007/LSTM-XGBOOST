import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib.colors as mcolors
import colorsys

def adjust_lightness(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)  # 转换颜色到RGB
    h, l, s = colorsys.rgb_to_hls(*c)  # 使用colorsys库转换到HLS
    new_c = colorsys.hls_to_rgb(h, max(0, min(1, l * amount)), s)  # 调整亮度并转回RGB
    return mcolors.to_hex(new_c)  # 将RGB颜色转换为十六进制颜色

def single_scatter_with_envelope(series, base_color, marker, model_name='LSTM', set='Val', save_path=None):
    """
    series: pd.Series，索引为X，值为Y，索引和值都需要有名称(x_label,y_label)
    """
    x = series.index
    y = series.values

    # Sort data for fitting
    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    sorted_y = y[sorted_indices]

    # Create a smooth curve using cubic spline interpolation
    model = make_interp_spline(sorted_x, sorted_y)
    xs = np.linspace(min(sorted_x), max(sorted_x), 300)
    ys = model(xs)

    # Calculate lighter and darker colors
    lighter_color = adjust_lightness(base_color, amount=1.2)  # 更亮的颜色
    darker_color = adjust_lightness(base_color, amount=0.8)   # 更暗的颜色

    # Plot scatter plot and envelope line
    plt.scatter(x, y, color=lighter_color, marker=marker, label=f'Data Points of {model_name} in {set} Set')
    plt.plot(xs, ys, color=darker_color, label=f'Envelope Line of {model_name} in {set} Set')

    plt.xlabel(series.index.name if series.index.name else 'Index')
    plt.ylabel(series.name if series.name else 'Value')
    plt.title(f'{series.name} of {model_name} Model on {set} Set by Different {series.index.name}')
    plt.legend()
    # 网格线调整为虚线
    plt.grid(True, linestyle='--')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def comparison_scatter_with_envelope(data1, data2, title='Scatter Plot with Envelope Lines', xlabel='T', ylabel='R2'):
    """
    """

    # Plot settings
    plt.figure(figsize=(10, 5))
    single_scatter_with_envelope(data1, 'blue', 'o', 'LSTM')
    single_scatter_with_envelope(data2, 'green', 'x', 'XGB/FLIXNet')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    # 保存
    # plt.savefig('comparison_scatter_with_envelope.png')
    plt.show()

