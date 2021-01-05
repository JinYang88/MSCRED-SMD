import os
import math
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from utils.matrix_generator import numbers
from model.mscred import MSCRED
from utils.data import load_data


def train(DataLoader, model, Optimizer, epochs, Device):
    model = model.to(Device)
    print("------training on {}-------".format(Device))
    for epoch in range(epochs):
        train_l_sum, n = 0.0, 0
        for x in tqdm(DataLoader):
            x = x.to(Device)
            x = x.squeeze()
            # print(type(x))
            l = torch.mean((model(x) - x[-1].unsqueeze(0)) ** 2)
            train_l_sum += l
            Optimizer.zero_grad()
            l.backward()
            Optimizer.step()
            n += 1

        print("[Epoch %d/%d] [loss: %f]" % (epoch + 1, epochs, train_l_sum / n))


def test(DataLoader, model, number):
    print("------Testing-------")
    get_start_end_path = './SMD/data_concat/data-' + number + '.csv'
    data_get_len = len(np.array(pd.read_csv(get_start_end_path, header=None), dtype=np.float64)) // 2
    index = math.ceil(data_get_len / 10)
    if not os.path.exists('./data/matrix_data_SMD-' + number + '/reconstructed_data'):
        os.makedirs('./data/matrix_data_SMD-' + number + '/reconstructed_data')
    reconstructed_data_path = './data/matrix_data_SMD-' + number + '/reconstructed_data'
    with torch.no_grad():
        for x in DataLoader:
            x = x.to(device)
            x = x.squeeze()
            reconstructed_matrix = model(x)
            path_temp = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(index) + ".npy")
            np.save(path_temp, reconstructed_matrix.cpu().detach().numpy())
            index += 1


if __name__ == '__main__':
    device = torch.device("cuda:0")
    print("device is", device)
    mscred = MSCRED(3, 256)

    # 训练阶段
    for num in numbers:

        dataLoader = load_data(num)
        optimizer = torch.optim.Adam(mscred.parameters(), lr=0.0002)
        train(dataLoader["train"], mscred, optimizer, 1, device)
        print("保存 %s 的模型" % num)

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')

        torch.save(mscred.state_dict(), './checkpoints/model' + num + '.pth')

    # 测试阶段
        mscred.load_state_dict(torch.load('./checkpoints/model' + num + '.pth'))
        mscred.to(device)
        test(dataLoader["test"], mscred, num)
        print('训练 %s 完成' % num)
