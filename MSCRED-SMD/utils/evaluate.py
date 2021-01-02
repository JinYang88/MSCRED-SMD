import numpy as np
import argparse
import matplotlib.pyplot as plt
# import string
import pandas as pd
# import math
import os
import torch

parser = argparse.ArgumentParser(description='MSCRED evaluation')
parser.add_argument('--thred_broken', type=int, default=0.005,
                    help='broken pixel thred')
parser.add_argument('--alpha', type=int, default=0.5,
                    help='scale coefficient of max valid anomaly')
parser.add_argument('--valid_start_point', type=int, default=23700,
                    help='test start point')
parser.add_argument('--valid_end_point', type=int, default=30000,
                    help='test end point')
parser.add_argument('--test_start_point', type=int, default=23700,
                    help='test start point')
parser.add_argument('--test_end_point', type=int, default=47388,
                    help='test end point')
parser.add_argument('--gap_time', type=int, default=10,
                    help='gap time between each segment')
parser.add_argument('--matrix_data_path', type=str, default='../data/matrix_data_SMD-1-2/',
                    help='matrix data path')

args = parser.parse_args()
print(args)

thred_b = args.thred_broken
alpha = args.alpha
gap_time = args.gap_time
valid_start = args.valid_start_point // gap_time
valid_end = args.valid_end_point // gap_time
test_start = args.test_start_point // gap_time
test_end = args.test_end_point // gap_time

valid_anomaly_score = np.zeros((valid_end - valid_start, 1))
test_anomaly_score = np.zeros((test_end - test_start, 1))

matrix_data_path = args.matrix_data_path
test_data_path = matrix_data_path + "test_data/"
reconstructed_data_path = matrix_data_path + "reconstructed_data/"
# reconstructed_data_path = matrix_data_path + "matrix_pred_data/"
criterion = torch.nn.MSELoss()

for i in range(valid_start, test_end):
    path_temp_1 = os.path.join(test_data_path, "test_data_" + str(i) + '.npy')
    gt_matrix_temp = np.load(path_temp_1)

    path_temp_2 = os.path.join(reconstructed_data_path, "reconstructed_data_" + str(i) + '.npy')
    # path_temp_2 = os.path.join(reconstructed_data_path, "pcc_matrix_full_test_" + str(i) + '_pred_output.npy')
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
    if i < valid_end:
        valid_anomaly_score[i - valid_start] = num_broken

    test_anomaly_score[i - test_start] = num_broken

valid_anomaly_max = np.max(valid_anomaly_score.ravel())
test_anomaly_score = test_anomaly_score.ravel()
# print(test_anomaly_score)
# plot anomaly score curve and identification result

tag_data_path = '../SMD/test_label/machine-1-2.csv'

data = np.array(pd.read_csv(tag_data_path, header=None), dtype=np.int)
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

fig, axes = plt.subplots()
test_num = test_end - test_start
plt.plot(test_anomaly_score, color='black', linewidth=2)
#plt.text(x=2, y=200, s='precision 80% recall 50% F1-score 61.5%', fontdict=dict(fontsize=20, color='black', family='monospace'))
threshold = np.full(test_num, valid_anomaly_max * alpha)
# print(threshold)

axes.plot(threshold, color='black', linestyle='--', linewidth=2)

# 增加异常区间并标记
for k in range(0, len(anomaly_pos)):
    axes.axvspan(anomaly_pos[k][0] / gap_time, anomaly_pos[k][1] / gap_time, color='gray', linewidth=2)

plt.xlabel('Test Time', fontsize=20)
plt.ylabel('Anomaly Score', fontsize=20)
axes.spines['right'].set_visible(True)
axes.spines['top'].set_visible(True)
axes.yaxis.set_ticks_position('left')
axes.xaxis.set_ticks_position('bottom')
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(left=0.2)
plt.title("MSCRED", size=20)

if not os.path.exists("../outputs"):
    os.makedirs("../outputs")
plt.savefig('../outputs/anomaly_score.jpg')
plt.show()
