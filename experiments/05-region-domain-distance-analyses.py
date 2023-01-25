import pandas as pd
import numpy as np

def get_distance(x, y):
    a = np.cos(x['col32'].to_numpy() - y['col32'].to_numpy())
    b = np.cos(x['col33'].to_numpy()) * np.cos(y['col33'].to_numpy())
    c = np.sin(x['col33'].to_numpy()) * np.sin(y['col33'].to_numpy())
    return np.arccos(a * b + c)

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    print(df)
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9
    df['col32'] = df['col32'] * np.pi / 180
    df['col33'] = df['col33'] * np.pi / 180

    distance = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        # print(df[i:i+1].to_numpy(), df.to_numpy())
        distance[i] = get_distance(df, df[i:i+1])
    
    # find the top 1th nearest neighbors
    top_5th = np.argsort(distance, axis=1)[:, 5]
    top_5th_distance = distance[np.arange(len(df)), top_5th] * 6371
    print(top_5th)
    print(np.sort(top_5th_distance))