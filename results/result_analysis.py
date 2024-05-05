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

def single_scatter_with_envelope(series, base_color, marker, model_name='LSTM', set='Val', save_path=None,show=True):
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
    plt.plot(xs, ys, color=darker_color, label=f'Fitting Line of {model_name} in {set} Set')

    plt.xlabel(series.index.name if series.index.name else 'Index')
    plt.ylabel(series.name if series.name else 'Value')
    plt.title(f'{series.name} of {model_name} Model on {set} Set by Different {series.index.name}')
    plt.legend()
    # 网格线调整为虚线
    plt.grid(True, linestyle='--')
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()

def comparison_scatter_with_envelope(series1, series2, title='Scatter Plot with Fitting Lines', xlabel=None, ylabel=None, model_name1='Model 1', model_name2='Model 2',set='val', save=False, save_path='comparison_plot.png'):
    """
    在同一图表中绘制两个数据集的散点图和包络线，用以比较。

    参数:
    - series1, series2: pd.Series，两个数据集，索引为X，值为Y，需要有名称。
    - title: str, 图表的标题。
    - xlabel: str, X轴的标签，默认为series1索引的名称。
    - ylabel: str, Y轴的标签，默认为series1值的名称。
    - legend1, legend2: str, 两个数据集的图例标签。
    - save: bool, 是否将图表保存为图片。
    - save_path: str, 图片保存路径。
    """
    # 如果没有提供xlabel和ylabel，使用series1的索引名和值名
    if xlabel is None:
        xlabel = series1.index.name if series1.index.name else 'Index'
    if ylabel is None:
        ylabel = series1.name if series1.name else 'Value'

    plt.figure(figsize=(10, 6))
    # 绘制第一个数据集
    single_scatter_with_envelope(series1, 'blue', 'o', model_name=model_name1, set=set,show=False)
    # 绘制第二个数据集
    single_scatter_with_envelope(series2, 'green', 'x', model_name=model_name2,set=set,show=False)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--')
    plt.legend()

    if save:
        plt.savefig(save_path, dpi=300)
    plt.show()