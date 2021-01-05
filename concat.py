import pandas as pd
import os

# concatenate train data and test data

train_dir = './SMD/train/'
test_dir = './SMD/test/'


def concat_csv(train_path, test_path):
    for root, dirs, files in os.walk(train_path):
        for name in files:
            path_temp = os.path.join(root, name)
            path_temp = path_temp.replace('\\', '/')
            is_csv = path_temp[-4:-1] + path_temp[-1] == '.csv'
            train_data_path = train_path + name
            test_data_path = test_path + name
            if is_csv:
                number = name[8:-4]

                f1 = pd.read_csv(train_data_path, header=None)
                print('First file is' + train_data_path)
                f2 = pd.read_csv(test_data_path, header=None)
                print('Second file is' + test_data_path)
                two_file_list = [f1, f2]
                file = pd.concat(two_file_list, axis=0)
                path_saved = './SMD/data_concat/'
                if not os.path.exists(path_saved):
                    os.mkdir(path_saved)
                file.to_csv(path_saved + 'data-' + number + '.csv', sep=',', header=False, index=False)
