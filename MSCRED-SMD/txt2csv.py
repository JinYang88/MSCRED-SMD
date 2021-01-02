import csv

csvFile = open('./SMD/test_label/machine-1-2.csv', 'w+', newline='')
writer = csv.writer(csvFile)
csvRow = []

f = open('./SMD/test_label/machine-1-2.txt', 'r', encoding='utf-8')
for line in f:
    csvRow = line.split(',')
    writer.writerow(csvRow)

f.close()
csvFile.close()