import pandas as pd
import geopandas
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    
    # calculate the average temperature
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    # Fahrenheit scale to Celsius scale
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9
    df['avg'] = df[temperature_column].mean(axis=1)
    df['std'] = df[temperature_column].std(axis=1)
    df['max'] = df[temperature_column].max(axis=1)
    df['min'] = df[temperature_column].min(axis=1)
    df['range'] = df['max'] - df['min']
    df['index'] = df.index
    
    print(max(df['avg']), min(df['avg'])) # 25.913978494623645 -7.347670250896054
    print(max(df['std']), min(df['std'])) # 8.274704151406832 0.7664302788653442
    print(max(df['max']), min(df['max'])) # 28.333333333333332 2.2222222222222223
    print(max(df['min']), min(df['min'])) # 24.444444444444443 -22.77777777777778

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    
    # We can now plot our ``GeoDataFrame``.
    # split the data into 6 parts
    color_order = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    for index in ['avg', 'std', 'max', 'min', 'range', 'col1', 'col31', 'index']:
        max_val = max(df[index])
        min_val = min(df[index])
        step = (max_val - min_val) / 6
        ax = world[world.name == 'United States of America'].plot(color='white', edgecolor='black')
        for i in range(6):
            sub_df = df[(df[index] >= min_val + step * i) & (df[index] < min_val + step * (i + 1))]
            gdf = geopandas.GeoDataFrame(sub_df, geometry=geopandas.points_from_xy(sub_df.col33, sub_df.col32))
            gdf.plot(ax=ax, color=color_order[i], markersize=1, label=f'{min_val + step * i:.2f} - {min_val + step * (i + 1):.2f}')
        plt.legend()
        plt.title(index)
        plt.savefig(f'{index}.png')
    
