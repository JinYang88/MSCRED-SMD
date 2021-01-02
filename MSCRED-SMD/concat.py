import pandas as pd

f1 = pd.read_csv('./SMD/train/machine-1-2.csv', header=None)
f2 = pd.read_csv('./SMD/test/machine-1-2.csv', header=None)
file = [f1, f2]
train = pd.concat(file, axis=0)
train.to_csv('./SMD/data-1-2.csv', sep=',', header=False, index=False)
