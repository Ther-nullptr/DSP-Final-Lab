import numpy as np
import pandas as pd

def hampel_filter_forloop(array, window_size, n_sigmas=3, normalize=False):
    
    n = len(array)
    k = 1.4826 # scale factor for Gaussian distribution
    
    indices = []
    new_array = array.copy()

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
            new_array[i] = x0
    
    return indices, new_array

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df = df[(df['col32'] < 50) & (df['col32'] > 25)]
    # calculate the average temperature
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    # Fahrenheit scale to Celsius scale
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9

    data = df[1:2][['col' + str(i) for i in range(1, 32)]].to_numpy().squeeze()
    data[21] += 10
    smooth_data = hampel_filter_forloop(data, 3)[1]
    print(data)
    print(smooth_data)