import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Signature Matrix Generator')
parser.add_argument('--step_max', type=int, default=5,
                    help='maximum step in ConvLSTM')
parser.add_argument('--gap_time', type=int, default=10,
                    help='gap time between each segment')
parser.add_argument('--win_size', type=int, default=[10, 30, 60],
                    help='window size of each segment')

args = parser.parse_args()

step_max = args.step_max
gap_time = args.gap_time
win_size = args.win_size

data_dir_path = 'D:/MSCRED-SMD/SMD/data_concat/'
csvs = os.listdir(data_dir_path)

csv_path = []

for i in csvs:
    csv_path.append(data_dir_path + i)

numbers = []

for j in csvs:
    name_temp = os.path.split(j)[1]
    numbers.append(name_temp[5:-4])

save_path = '../data/'


def generate_signature_matrix_node(data_path):
    data = np.array(pd.read_csv(data_path, header=None), dtype=np.float64)
    data = data.transpose()
    sensor_n = data.shape[0]
    length = data.shape[1]
    # min-max normalization
    max_value = np.max(data, axis=1)
    min_value = np.min(data, axis=1)
    data = (np.transpose(data) - min_value) / (max_value - min_value + 1e-6)
    data = np.transpose(data)

    name = os.path.split(data_path)[1]
    num = name[5:-4]

    # multi-scale signature matrix generation
    for w in range(len(win_size)):
        matrix_all = []
        win = win_size[w]
        for t in range(0, length, gap_time):
            # print t
            matrix_t = np.zeros((sensor_n, sensor_n))
            if t >= 60:
                for l in range(sensor_n):
                    for m in range(l, sensor_n):
                        # if np.var(data[i, t - win:t]) and np.var(data[j, t - win:t]):
                        matrix_t[l][m] = np.inner(data[l, t - win:t], data[m, t - win:t]) / win  # rescale by win
                        matrix_t[m][l] = matrix_t[l][m]
            matrix_all.append(matrix_t)

        is_csv = data_path[-4:-1] + data_path[-1] == '.csv'
        if is_csv:

            matrix_data_path = save_path + "matrix_data_SMD-" + num + '/'

            if not os.path.exists(matrix_data_path):
                os.makedirs(matrix_data_path)
            path_temp = matrix_data_path + "matrix_win_" + str(win)
            np.save(path_temp, matrix_all)
            del matrix_all[:]

    print('Generation for ' + name + ' complete')


def generate_train_test_data(number):
    # data sample generation
    print("generating train/test data samples of " + number)
    matrix_data_path = save_path + "matrix_data_SMD-" + number + '/'

    get_start_end_path = '../SMD/data_concat/data-' + number + '.csv'
    data_get_len = len(np.array(pd.read_csv(get_start_end_path, header=None), dtype=np.float64))

    train_start = 0
    train_end = data_get_len // 2
    test_start = train_end
    test_end = data_get_len

    train_data_path = matrix_data_path + "train_data/"
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    test_data_path = matrix_data_path + "test_data/"
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    data_all = []

    for w in range(len(win_size)):
        path_temp = matrix_data_path + "matrix_win_" + str(win_size[w]) + ".npy"
        data_all.append(np.load(path_temp))

    train_test_time = [[train_start, train_end], [test_start, test_end]]
    for m in range(len(train_test_time)):
        for data_id in range(int(train_test_time[m][0] / gap_time), int(train_test_time[m][1] / gap_time)):
            # print data_id
            step_multi_matrix = []
            for step_id in range(step_max, 0, -1):
                multi_matrix = []
                for k in range(len(win_size)):
                    multi_matrix.append(data_all[k][data_id - step_id])
                step_multi_matrix.append(multi_matrix)

            if (train_start / gap_time + win_size[-1] / gap_time + step_max) <= data_id < (
                    train_end / gap_time):  # remove start points with invalid value
                path_temp = os.path.join(train_data_path, 'train_data_' + str(data_id))
                np.save(path_temp, step_multi_matrix)
            elif (test_start / gap_time) <= data_id < (test_end / gap_time):
                path_temp = os.path.join(test_data_path, 'test_data_' + str(data_id))
                np.save(path_temp, step_multi_matrix)

            del step_multi_matrix[:]
    print("train/test data generation finish!")


if __name__ == '__main__':
    for path in csv_path:
        generate_signature_matrix_node(path)
    for numb in numbers:
        generate_train_test_data(numb)
