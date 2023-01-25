import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_distance(x, y):
    a = np.cos(x['col33'].to_numpy() - y['col33'].to_numpy())
    b = np.cos(x['col32'].to_numpy()) * np.cos(y['col32'].to_numpy())
    c = np.sin(x['col32'].to_numpy()) * np.sin(y['col32'].to_numpy())
    return np.arccos(a * b + c)


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9
    df['col32'] = df['col32'] * np.pi / 180
    df['col33'] = df['col33'] * np.pi / 180

    df['distance'] = get_distance(df, df[22:23]) * 6371
    # sort the df
    df = df.sort_values(by='distance', ascending=True)
    df['col32'] = df['col32'] / np.pi * 180
    df['col33'] = df['col33'] / np.pi * 180
    avg_data = 0
    exp_avg_data = 0
    temp = 50

    for i in range(5):
        data = df[i:i + 1][temperature_column].to_numpy().squeeze()
        plt.plot(data,
                 label='distance: {:.2f} km, location: {:.2f},{:.2f}'.format(
                     df[i:i + 1]['distance'].to_numpy().squeeze(),
                     df[i:i + 1]['col32'].to_numpy().squeeze(),
                     df[i:i + 1]['col33'].to_numpy().squeeze()),
                 marker='o')

        if i > 0:
        # normal avg
            avg_data += data
        # exp average
            if i == 1:
                exp_avg_data = data
                exp_sum = np.exp(-df[i:i + 1]['distance'].to_numpy().squeeze() / temp) + np.exp(-df[i:i + 1]['distance'].to_numpy().squeeze() / temp)
            else:
                exp_now = np.exp(-df[i:i + 1]['distance'].to_numpy().squeeze() / temp)
                exp_avg_data = exp_avg_data * exp_sum / (exp_sum + exp_now) + data * exp_now / (exp_sum + exp_now)
                exp_sum += exp_now

    avg_data = avg_data / 4
    plt.plot(avg_data, label='average', marker='o')
    plt.plot(df[0:1][temperature_column].to_numpy().squeeze() - avg_data,
             label='average diff',
             marker='o')
    plt.plot(exp_avg_data, label='exp_average', marker='o')
    plt.plot(df[0:1][temperature_column].to_numpy().squeeze() - exp_avg_data,
             label='exp_average diff',
             marker='o')
    plt.legend()
    plt.show()