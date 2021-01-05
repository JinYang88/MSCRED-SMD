import csv
import os

train_path = './SMD/train/'
test_path = './SMD/test/'
test_label_path = './SMD/test_label/'
path = [train_path, test_path, test_label_path]


def txt2csv(path_root=None):
    for path_k in path_root:
        for root, dirs, files in os.walk(path_k):
            for name in files:
                name_without_suffix = os.path.splitext(name)[0]
                path_temp = os.path.join(root, name)
                path_temp = path_temp.replace('\\', '/')
                is_txt = path_temp[-4:-1] + path_temp[-1] == '.txt'
                if is_txt:
                    print('Begin generating .csv type using raw .txt:' + path_temp)
                    csvFile = open(path_k + name_without_suffix + '.csv', 'w+', newline='')
                    writer = csv.writer(csvFile)

                    f = open(path_temp, 'r', encoding='utf-8')
                    for line in f:
                        csvRow = line.split(',')
                        writer.writerow(csvRow)

                    f.close()
                    csvFile.close()
