import pandas as pd
import geopandas
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load the data from csv
    df = pd.read_csv('data.csv')
    print(df)
    # read the Latitude and Longitude columns as a geometry
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.col33, df.col32))
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    ax = world[world.name == 'United States of America'].plot(color='white', edgecolor='black')
    # We can now plot our ``GeoDataFrame``.
    gdf.plot(ax=ax, color='purple', markersize=1)
    plt.savefig('plot.png')

