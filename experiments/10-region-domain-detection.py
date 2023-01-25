import pandas as pd
import numpy as np
from tqdm import trange
import copy

def hampel_filter_forloop(array, window_size, n_sigmas=10, normalize=False):
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


def get_distance(x, y):
    a = np.cos(x['col33'].to_numpy() - y['col33'].to_numpy())
    b = np.cos(x['col32'].to_numpy()) * np.cos(y['col32'].to_numpy())
    c = np.sin(x['col32'].to_numpy()) * np.sin(y['col32'].to_numpy())
    return np.arccos(a * b + c)


if __name__ == '__main__':
    # hyper parameters
    temp = 100
    point_num = 4
    diff_threshold = 0
    window_length = 11
    distance_threshold = 10000
    delta_temperature = 20

    df = pd.read_csv('data.csv')
    df = df.reset_index()
    # calculate the average temperature
    temperature_column = ['col' + str(i) for i in range(1, 32)]
    # Fahrenheit scale to Celsius scale
    df[temperature_column] = (df[temperature_column] - 32) * 5 / 9

    df['col32'] = df['col32'] * np.pi / 180
    df['col33'] = df['col33'] * np.pi / 180

    normal_value = -np.ones(len(df))
    normal_value_idx = -np.ones(len(df))
    neighbor_dict = {}

    # add the keys of neighbor_dict
    for i in range(len(df)):
        neighbor_dict[i] = []

    for i in range(len(df)):
        data = df[i:i + 1][temperature_column].to_numpy().squeeze()
        df['distance'] = get_distance(df, df[i:i + 1]) * 6371
        sorted_df = df[df['distance'] < distance_threshold]
        new_df = sorted_df.sort_values(by='distance', ascending=True)
        
        # get the top n results
        new_df = new_df[1:1 + point_num]
        # update the dictionary

        for index in new_df.index.to_numpy().squeeze():
            if index in neighbor_dict:
                neighbor_dict[index].append(i)
            else:
                neighbor_dict[index] = [i]
        
        ref_data = new_df[temperature_column].to_numpy().squeeze()
        ref_distance = new_df['distance'].to_numpy().squeeze()

        # get the averaged data
        exp_avg_data = 0
        for j in range(len(ref_data)):
            smooth_data = hampel_filter_forloop(ref_data[j], window_length)[1]
            if j == 0:
                exp_avg_data = smooth_data
                exp_sum = np.exp(-ref_distance[j] / temp)
            else:
                exp_now = np.exp(-ref_distance[j] / temp)
                exp_avg_data = exp_avg_data * exp_sum / (exp_sum + exp_now) + smooth_data * exp_now / (exp_sum + exp_now)
                exp_sum += exp_now

        diff = exp_avg_data - data
        diff -= np.mean(diff)
        max_val = np.max(np.abs(diff))
        mean = (np.sum(np.abs(diff)) - max_val) / (len(diff) - 1)
        max_diff = max_val / mean
        max_diff_pos = np.argmax(np.abs(diff))
        
        if max_diff > diff_threshold:
            normal_value[i] = max_diff
            normal_value_idx[i] = max_diff_pos

    success_num = 0
    total_num = 0

    print(normal_value)
    print(normal_value_idx)
    print(df)
    print(neighbor_dict)

    with trange(len(df)) as t:
        for i in t:
            for j in range(31):
                total_num += 1
                # add the abnormal value
                new_df = copy.deepcopy(df)
                new_df.loc[i:i, f'col{j + 1}'] += delta_temperature
                data = new_df[i:i + 1][temperature_column].to_numpy().squeeze()
                new_max_value = copy.deepcopy(normal_value)
                new_max_value_idx = copy.deepcopy(normal_value_idx)
                # print(neighbor_dict[i])

                for k in range(len(df)):
                    # we must calculate it again
                    if (k in neighbor_dict[i]):
                        new_df['distance'] = get_distance(new_df, new_df[k:k + 1]) * 6371
                        sorted_df = new_df[new_df['distance'] < distance_threshold]
                        new_df_2 = sorted_df.sort_values(by='distance', ascending=True)
                        
                        # get the top n results
                        ref_df = new_df_2[1:1 + point_num]
                        
                        ref_data = ref_df[temperature_column].to_numpy().squeeze()
                        ref_distance = ref_df['distance'].to_numpy().squeeze()

                        # get the averaged data
                        exp_avg_data = 0
                        for l in range(point_num):
                            smooth_data = hampel_filter_forloop(ref_data[l], window_length)[1]
                            if l == 0:
                                exp_avg_data = smooth_data
                                exp_sum = np.exp(-ref_distance[l] / temp)
                            else:
                                exp_now = np.exp(-ref_distance[l] / temp)
                                exp_avg_data = exp_avg_data * exp_sum / (exp_sum + exp_now) + smooth_data * exp_now / (exp_sum + exp_now)
                                exp_sum += exp_now

                        diff = exp_avg_data - data
                        diff -= np.mean(diff)
                        max_val = np.max(np.abs(diff))
                        mean = (np.sum(np.abs(diff)) - max_val) / (len(diff) - 1)
                        max_diff = max_val / mean
                        max_diff_pos = np.argmax(np.abs(diff))

                        if max_diff > diff_threshold:
                            new_max_value[k] = max_diff
                            new_max_value_idx[k] = max_diff_pos 
                
                new_df['distance'] = get_distance(new_df, new_df[i:i + 1]) * 6371
                sorted_df = new_df[new_df['distance'] < distance_threshold]
                new_df_2 = sorted_df.sort_values(by='distance', ascending=True)
                
                # get the top n results
                ref_df = new_df_2[1:1 + point_num]
                
                ref_data = ref_df[temperature_column].to_numpy().squeeze()
                ref_distance = ref_df['distance'].to_numpy().squeeze()

                # get the averaged data
                for l in range(point_num):
                    smooth_data = hampel_filter_forloop(ref_data[l], window_length)[1]
                    if l == 0:
                        exp_avg_data = smooth_data
                        exp_sum = np.exp(-ref_distance[l] / temp)
                    else:
                        exp_now = np.exp(-ref_distance[l] / temp)
                        exp_avg_data = exp_avg_data * exp_sum / (exp_sum + exp_now) + smooth_data * exp_now / (exp_sum + exp_now)
                        exp_sum += exp_now

                diff = exp_avg_data - data
                diff -= np.mean(diff)
                max_val = np.max(np.abs(diff))
                mean = (np.sum(np.abs(diff)) - max_val) / (len(diff) - 1)
                max_diff = max_val / mean
                max_diff_pos = np.argmax(np.abs(diff))
                # print(max_diff)

                if max_diff > diff_threshold:
                    new_max_value[i] = max_diff
                    new_max_value_idx[i] = max_diff_pos 
                
                # print(new_max_value_idx[i], j, new_max_value[i], np.max(new_max_value))
                if (new_max_value_idx[i] == j) and new_max_value[i] == np.max(new_max_value):
                    success_num += 1
            
            t.set_postfix({'acc': success_num / total_num})

    print('success_num: ', success_num)
    print('total_num: ', total_num)
    print('acc: ', success_num / total_num)
