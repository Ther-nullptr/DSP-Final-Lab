import pandas as pd
import matplotlib.pyplot as plt
import geopandas

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    color_order = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    # calculate the difference between k days
    for k in [1, 5, 10]:
        diff_df = df[temperature_column].diff(periods=k, axis=1)
        diff_df = diff_df.fillna(0)
        diff_df['max'] = diff_df.abs().max(axis=1)
        diff_df['col32'] = df['col32']
        diff_df['col33'] = df['col33']

        # plot the data
        max_val = max(diff_df['max'])
        min_val = min(diff_df['max'])
        step = (max_val - min_val) / 6
        ax = world[world.name == 'United States of America'].plot(color='white', edgecolor='black')
        for i in range(6):
            sub_df = diff_df[(diff_df['max'] >= min_val + step * i) & (diff_df['max'] < min_val + step * (i + 1))]
            gdf = geopandas.GeoDataFrame(sub_df, geometry=geopandas.points_from_xy(sub_df.col33, sub_df.col32))
            gdf.plot(ax=ax, color=color_order[i], markersize=1, label=f'{min_val + step * i:.2f} - {min_val + step * (i + 1):.2f}')
        plt.legend()
        plt.title(f'diff_{k}')
        plt.savefig(f'diff_{k}.png')
    