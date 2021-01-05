import numpy as np
import argparse
import pandas as pd
import os
import torch
from sklearn.metrics import f1_score
from utils.matrix_generator import numbers

parser = argparse.ArgumentParser(description='MSCRED evaluation')
parser.add_argument('--thred_broken', type=int, default=0.005,
                    help='broken pixel thred')
parser.add_argument('--gap_time', type=int, default=10,
                    help='gap time between each segment')

args = parser.parse_args()
thred_b = args.thred_broken
gap_time = args.gap_time


def adjust_predicts(score, label, percent=None,
                    pred=None,
                    threshold=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is higher than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
        :param pred:
        :param label:
        :param score:
        :param calc_latency:
        :param threshold:
        :param percent:
    """
    if score is not None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    predict = []
    if pred is None:
        if percent is not None:
            threshold = np.percentile(score, percent)
            predict = score > threshold
        elif threshold is not None:
            predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for k in range(len(predict)):
        if actual[k] and predict[k] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(k, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[k]:
            anomaly_state = False
        if anomaly_state:
            predict[k] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def __iter_thresholds_without_adjust(score, label):
    best_f1 = -float("inf")
    best_theta = None
    for anomaly_ratio in np.linspace(1e-4, 10, 500):
        pred = anomaly_ratio < score

        f1 = f1_score(pred, label)
        if f1 > best_f1:
            best_f1 = f1
            best_theta = anomaly_ratio
    return best_f1, best_theta


def __iter_thresholds(score, label):
    best_f1 = -float("inf")
    best_theta = None
    best_adjust = None
    for anomaly_ratio in np.linspace(1e-3, 0.3, 50):
        adjusted_anomaly = adjust_predicts(
            score, label, percent=100 * (1 - anomaly_ratio)
        )
        f1 = f1_score(adjusted_anomaly, label)
        if f1 > best_f1:
            best_f1 = f1
            best_adjust = adjusted_anomaly
            best_theta = anomaly_ratio
    return best_f1, best_theta, best_adjust


def evaluate(number):
    get_start_end_path = 'D:/MSCRED-SMD/SMD/data_concat/data-' + number + '.csv'
    data_get_len = len(np.array(pd.read_csv(get_start_end_path, header=None), dtype=np.float64)) // 2
    test_start = (data_get_len + 10 - data_get_len % 10) // 10
    test_end = (2 * data_get_len) // 10
    valid_start = test_start
    valid_end = valid_start + 5000

    valid_anomaly_score = np.zeros((valid_end - valid_start, 1))
    test_anomaly_score = np.zeros((test_end - test_start, 1))

    matrix_data_path = '../data/' + "matrix_data_SMD-" + number + '/'
    test_data_path = matrix_data_path + "test_data/"
    reconstructed_data_path = matrix_data_path + "reconstructed_data/"
    criterion = torch.nn.MSELoss()

    for m in range(valid_start, test_end):
        path_temp_1 = os.path.join(test_data_path, "test_data_" + str(m) + '.npy')
        gt_matrix_temp = np.load(path_temp_1)

        path_temp_2 = os.path.join(reconstructed_data_path, "reconstructed_data_" + str(m) + '.npy')
        reconstructed_matrix_temp = np.load(path_temp_2)
        # reconstructed_matrix_temp = np.transpose(reconstructed_matrix_temp, [0, 3, 1, 2])
        # print(reconstructed_matrix_temp.shape)
        # first (short) duration scale for evaluation
        select_gt_matrix = np.array(gt_matrix_temp)[-1][0]  # get last step matrix

        select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]

        # compute number of broken element in residual matrix
        select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
        num_broken = len(select_matrix_error[select_matrix_error > thred_b])

        # print num_broken
        if m < valid_end:
            valid_anomaly_score[m - valid_start] = num_broken

        test_anomaly_score[m - test_start] = num_broken

    test_anomaly_score = test_anomaly_score.ravel()
    # print(test_anomaly_score)
    # plot anomaly score curve and identification result

    anomaly_score = []

    for m in range(len(test_anomaly_score)):
        for j in range(10):
            anomaly_score.append(test_anomaly_score[m])

    tag_data_path = '../SMD/test_label/machine-' + number + '.csv'

    data = np.array(pd.read_csv(tag_data_path, header=None), dtype=np.int)
    tag = data[10 - data_get_len % 10:10 - data_get_len % 10 + len(anomaly_score)]
    anomaly_pos = []

    key = 0
    while key < data.size:
        if data[key] != 1:
            key += 1
        else:
            start = key
            while data[key] != 0:
                key += 1
            end = key
            anomaly_pos.append([start, end])
    return __iter_thresholds(anomaly_score, tag)[0], __iter_thresholds_without_adjust(anomaly_score, tag)[0]


index = []
f1_with_adjust = []
f1_without_adjust = []

for i in numbers:
    index.append(i)
    f1_with_adjust.append(round(evaluate(i)[0], 2))
    f1_without_adjust.append(round(evaluate(i)[1], 2))
    print('Evaluate %s finished' % i)

csvFile = open('D:/MSCRED-SMD/performance.csv', 'w+', newline='')
f1_data = pd.DataFrame({'Series number': index, 'F1 with adjust': f1_with_adjust,
                        'F1 without adjust': f1_without_adjust})
f1_data.set_index('Series number')
f1_data.to_csv(csvFile)
