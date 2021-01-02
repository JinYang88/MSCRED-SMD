import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tag_data_path = './SMD/test_label/machine-1-2.csv'

data = np.array(pd.read_csv(tag_data_path, header=None), dtype=np.int)
index = []

key = 0
while key < data.size:
    if data[key] != 1:
        key += 1
    else:
        start = key
        while data[key] != 0:
            key += 1
        end = key
        index.append([start, key])

print(index)

fig, axes = plt.subplots()
plt.plot(data, color='black', linewidth=2)
plt.xlabel('Test Time', fontsize=20)
plt.ylabel('Anomaly Score', fontsize=20)
axes.spines['right'].set_visible(True)
axes.spines['top'].set_visible(True)
axes.yaxis.set_ticks_position('left')
axes.xaxis.set_ticks_position('bottom')
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(left=0.2)
plt.title("MSCRED", size=20)
plt.show()
