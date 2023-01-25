import pandas as pd
import matplotlib.pyplot as plt
import geopandas
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    
    # calculate the average temperature
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    # Fahrenheit scale to Celsius scale
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9
    
    region_data = {
        'north': df[22:23][temperature_column].to_numpy().squeeze(),
        'south': df[5:6][temperature_column].to_numpy().squeeze(),
        'north east': df[124:125][temperature_column].to_numpy().squeeze(),
        'north west': df[130:131][temperature_column].to_numpy().squeeze()
    }

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    # We can now plot our ``GeoDataFrame``.
    # split the data into 6 parts
    color_order = ['purple', 'blue', 'green']
    for index, data in region_data.items():
        # calculate mean square loss
        df['loss'] = np.sum((df[temperature_column] - data)**2, axis=1)
        df = df.sort_values(by=['loss'], ascending=True)
        new_df = df[0:30]
        max_val = max(new_df['loss'])
        min_val = min(new_df['loss'])
        step = (max_val - min_val) / 3

        ax = world[world.name == 'United States of America'].plot(color='white', edgecolor='black')
        for i in range(3):
            sub_new_df = new_df[(new_df['loss'] >= min_val + step * i) & (new_df['loss'] < min_val + step * (i + 1))]
            gnew_df = geopandas.GeoDataFrame(sub_new_df, geometry=geopandas.points_from_xy(sub_new_df.col33, sub_new_df.col32))
            gnew_df.plot(ax=ax, color=color_order[i], markersize=1, label=f'{min_val + step * i:.2f} - {min_val + step * (i + 1):.2f}')
        plt.legend()
        plt.title(index)
        plt.savefig(f'{index}.png')