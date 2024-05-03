# Modified version of the file published by Yaguang Li @ https://github.com/liyaguang/DCRNN

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df1, df2, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    df1 = df1.transpose()
    num_samples, num_nodes  = df1.shape
    data = np.expand_dims(df1.values, axis=-1)
    feature_list = [data]
   
    df2 = df2.transpose()
    num_samples, num_nodes  = df2.shape
    data = np.expand_dims(df2.values, axis=-1)
    feature_list.append(data)
    
    if add_time_in_day:
        day_of_month = (df1.index.values.astype("datetime64[D]") - df1.index.values.astype("datetime64[M]")) + 1
        day_of_month = day_of_month.astype(int)
        time_ind = (df1.index.values.astype("datetime64") - df1.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D") #percentage of the day
        time_ind = time_ind.astype(float)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))   
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = pd.to_datetime(df1.index.values).day_of_week
        dow = dow.astype(float)
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df1 = pd.read_csv(args.traffic_file_primary, index_col=0).fillna(value = 0)
    df2 = pd.read_csv(args.traffic_file_secondary, index_col=0).fillna(value = 0)    
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df1,
        df2,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="traffic_data/npz_files", help="Output directory.")
    parser.add_argument("--traffic_file_primary", type=str, default="traffic_data/pems_bay_speed.csv", help="Raw traffic readings - primary.",)
    parser.add_argument("--traffic_file_secondary", type=str, default="traffic_data/pems_bay_occupancy.csv", help="Raw traffic readings - secondary.",)
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
