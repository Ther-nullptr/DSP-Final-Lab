import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    # calculate the average temperature
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    # Fahrenheit scale to Celsius scale
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9

    region_dict = {}
    alaska_data = df[65:66][temperature_column].to_numpy().squeeze()
    region_dict['alaska'] = alaska_data
    hawaii_data = df[45:46][temperature_column].to_numpy().squeeze()
    region_dict['hawaii'] = hawaii_data
    northwest_data = df[130:131][temperature_column].to_numpy().squeeze()
    region_dict['northwest'] = northwest_data
    southwest_data = df[133:134][temperature_column].to_numpy().squeeze()
    region_dict['southwest'] = southwest_data
    north_data = df[22:23][temperature_column].to_numpy().squeeze()
    region_dict['north'] = north_data
    south_data = df[5:6][temperature_column].to_numpy().squeeze()
    region_dict['south'] = south_data
    northeast_data = df[124:125][temperature_column].to_numpy().squeeze()
    region_dict['northeast'] = northeast_data
    southeast_data = df[68:69][temperature_column].to_numpy().squeeze()
    region_dict['southeast'] = southeast_data

    df['max'] = df[temperature_column].max(axis=1)
    df['min'] = df[temperature_column].min(axis=1)
    df['range'] = df['max'] - df['min']

    # plot the data
    plt.figure()
    for region, data in region_dict.items():
        plt.plot(np.arange(0, 31), data, label=region, marker='o')
    plt.legend()
    plt.savefig('daily.png')

    # range over 30
    plt.figure()
    new_df = df[df['range'] > 30]
    for i in range(len(new_df)):
        data = new_df[i:i+1][temperature_column].to_numpy().squeeze()
        plt.plot(np.arange(0, 31), data, label=i, marker='o')
    plt.legend()
    plt.savefig('daily_range_30.png')

    # range between 25 and 30
    plt.figure()
    new_df = df[(df['range'] > 27) & (df['range'] <= 30)]
    for i in range(len(new_df)):
        data = new_df[i:i+1][temperature_column].to_numpy().squeeze()
        plt.plot(np.arange(0, 31), data, label=i, marker='o')
    plt.legend()
    plt.savefig('daily_range_27_30.png')