import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

def hampel_filter_forloop(array, window_size, n_sigmas=1, normalize=False):
    
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
    delta_temperature = 10
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
        data = df[i:i+1][temperature_column].to_numpy().squeeze()
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

    success_num = 0
    total_num = 0

    with trange(len(df)) as t:
        for i in t:
            for j in range(31):
                data = df[i:i+1][temperature_column].to_numpy().squeeze()
                data[j] += delta_temperature
                indices, factor = hampel_filter_forloop(data, window_length, normalize=normalize)
                total_num += 1
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
                        success_num += 1

            t.set_postfix({'acc': success_num / total_num})
    
    print(f'delete_max_min:{delete_max_min}, delta: {delta_temperature}, window: {window_length}, accuracy: {success_num / total_num}')

    # for i in range(len(df)):
    #     data = df[i:i+1][temperature_column].to_numpy().squeeze()
    #     data_with_delta = copy.deepcopy(data)
    #     data_with_delta[20] += delta_temperature
    #     indices, factor = hampel_filter_forloop(data_with_delta, 5)
    #     data_w_o_max_min = data[(data != data.max()) & (data != data.min())]
    #     std = data_w_o_max_min.std()
    #     print(f'{i}th data: {indices}, factor: {factor/std}')

    # plot the 114th data
    # data = df[0:1][temperature_column].to_numpy().squeeze() # hawaii
    # plt.figure()
    # plt.plot(np.arange(0, 31), data, label='114th', marker='o')
    # plt.legend()
    # plt.show()
    # smooth_data = df[45:46][temperature_column].to_numpy().squeeze() # hawaii
    # hard_data = df[22:23][temperature_column].to_numpy().squeeze() # north

    # smooth_data_with_delta = copy.deepcopy(smooth_data)
    # hard_data_with_delta = copy.deepcopy(hard_data)
    # smooth_data_with_delta[10] += delta_temperature
    # hard_data_with_delta[10] += delta_temperature

    # # plot its moving average
    # # plt.figure()
    # # plt.plot(np.arange(0, 31), smooth_data, label='hawaii', marker='o')
    # # plt.plot(np.arange(0, 31), hard_data, label='north', marker='o')
    # # plt.legend()
    # # plt.show()

    # _, idx1 = hampel_filter_forloop(smooth_data_with_delta, 3)
    # print(idx1)

    # _, idx2 = hampel_filter_forloop(hard_data_with_delta, 3)
    # print(idx2)
