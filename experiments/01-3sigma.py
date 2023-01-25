import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    delta_temperature = 20
    df = pd.read_csv('data.csv')
    # calculate the average temperature
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    # Fahrenheit scale to Celsius scale
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9

    df['avg'] = df[temperature_column].mean(axis=1)
    df['std'] = df[temperature_column].std(axis=1)

    abnormal_temperature_num = 0

    for j in range(len(df)):
        for item0 in temperature_column:
            # save the previous value
            prev_val = df.loc[j:j, item0].to_numpy().squeeze()
            # change the value
            df.loc[j:j, item0] = df.loc[j:j, item0] + delta_temperature
            print(f'set {j} row, {item0} column to {df[j:j+1][item0].to_numpy().squeeze()}')
            
            # detect the abnormal temperature
            for i in tqdm(range(len(df))):
                for item in temperature_column:
                    # delete the max val and min val to calculate the std
                    max_val = df[i:i+1][temperature_column].max(axis=1).to_numpy().squeeze()
                    min_val = df[i:i+1][temperature_column].min(axis=1).to_numpy().squeeze()
                    # delete the max val and min val and transform to numpy array
                    new_array = df[i:i+1][temperature_column].to_numpy().squeeze()
                    new_array = new_array[new_array != max_val]
                    new_array = new_array[new_array != min_val]
                    # calculate the std
                    new_std = new_array.std()
                    new_mean = new_array.mean()
                    
                    if (df[i:i+1][item].values > (new_mean + 3*new_std)) or (df[i:i+1][item].values < (new_mean - 3*new_std)):
                        abnormal_temperature_num += 1
                        print(f'abnormal temperature detected in {i} row, {item} column')
            # recover the value
            df.loc[i:i+1, item0] = prev_val
    
    print(abnormal_temperature_num)

