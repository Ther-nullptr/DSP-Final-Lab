import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import geopandas

def hampel_filter_forloop(array, window_size, n_sigmas=3, normalize=False):
    
    n = len(array)
    k = 1.4826 # scale factor for Gaussian distribution
    
    indices = []
    diff = []

    if normalize:
        x = np.arange(n)
        a, b = np.polyfit(x, array, 1)
        array = array - (a * x + b)
    
    for i in range(n):
        if i < window_size:
            x0 = np.median(array[0: window_size])
            mad = k * np.median(np.abs(array[0: window_size] - x0))
        elif i > (n - window_size):
            x0 = np.median(array[n - window_size:n])
            mad = k * np.median(np.abs(array[0: window_size] - x0))
        else:
            x0 = np.median(array[(i - window_size):(i + window_size)])
            mad = k * np.median(np.abs(array[(i - window_size):(i + window_size)] - x0))
        if (np.abs(array[i] - x0) > n_sigmas * mad):
            indices.append(i)
            diff.append((array[i] - x0))
    
    return indices, diff

if __name__ == '__main__':
    # hyper parameters
    delta_temperature = 20
    window_length = 11
    delete_max_min = True
    normalize = False

    df = pd.read_csv('data.csv')

    # calculate the average temperature
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    # Fahrenheit scale to Celsius scale
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9

    # calculate the normal value
    normal_value = -np.ones(len(df))
    normal_value_idx = -np.ones(len(df))

    for i in range(len(df)):
        data = df[i:i + 1][temperature_column].to_numpy().squeeze()
        indices, factor = hampel_filter_forloop(data, window_length, normalize=normalize)
        if len(factor) != 0:
            factor = np.array(factor)
            if delete_max_min:
                data_w_o_max_min = data[(data != data.max()) & (data != data.min())]
                std = data_w_o_max_min.std()
            else:
                std = data.std()
            factor = np.abs(factor / std)
            max_factor_idx = np.argmax(factor)
            max_idx = indices[max_factor_idx]
            normal_value[i] = factor[max_factor_idx]
            normal_value_idx[i] = max_idx

    success_num = np.zeros(len(df))

    with trange(len(df)) as t:
        for i in t:
            for j in range(31):
                data = df[i:i+1][temperature_column].to_numpy().squeeze()
                data[j] += delta_temperature
                indices, factor = hampel_filter_forloop(data, window_length, normalize=normalize)
                if len(factor) != 0:
                    factor = np.array(factor)

                    if delete_max_min:
                        data_w_o_max_min = data[(data != data.max()) & (data != data.min())]
                        std = data_w_o_max_min.std()
                    else:
                        std = data.std()

                    factor = np.abs(factor / std)
                    max_factor_idx = np.argmax(factor)
                    max_idx = indices[max_factor_idx]
                    if (factor[max_factor_idx] > np.max(normal_value)) and (max_idx == j):
                        success_num[i] += 1

    # merge the success num and df
    df['success_num'] = success_num
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    # split the data into 6 parts
    color_order = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    max_val = max(df['success_num'])
    min_val = min(df['success_num'])
    step = (max_val - min_val) / 6
    ax = world[world.name == 'United States of America'].plot(color='white', edgecolor='black')
    for i in range(6):
        sub_df = df[(df['success_num'] >= min_val + step * i) & (df['success_num'] < min_val + step * (i + 1))]
        gdf = geopandas.GeoDataFrame(sub_df, geometry=geopandas.points_from_xy(sub_df.col33, sub_df.col32))
        gdf.plot(ax=ax, color=color_order[i], markersize=1, label=f'{min_val + step * i:.2f} - {min_val + step * (i + 1):.2f}')
    plt.legend()
    plt.title('success_num')
    plt.savefig('success_num.png')
