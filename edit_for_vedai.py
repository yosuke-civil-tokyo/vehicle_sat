"""
You can start your training using VEDAI dataset editted by this file
"""
import numpy as np
import pandas as pd
import os
import glob
import shutil

data_path = 'custom_dataset/'
before = ''

os.makedirs(data_path + 'labels', exist_ok=True)
os.makedirs(data_path + 'img', exist_ok=True)

ann = pd.read_csv(data_path + 'annotation512.txt', delimiter=' ', header=None, dtype={'0':'str'})
def make_name(x):
    return '{0:08d}'.format(x) + '_co.png'
ann.iloc[:, 0] = ann.iloc[:, 0].map(make_name)

for i in range(len(ann)):
    if before != ann.iloc[i, 0]:
        f = open(data_path + 'train.txt', 'a')
        f.write(ann.iloc[i, 0] + '\n')
        f.close()

    before = ann.iloc[i, 0]

car_label = [1, 2, 3, 5, 9, 10, 12, 13]
car = ann[ann.iloc[:, -3].isin(car_label)]
car = car.iloc[:, [0, 4,5,6,7,8,9,10,11,12]].reset_index(drop=True)
car.iloc[:, -1] = 'car'

min_x = car.iloc[:, [1,2,3,4]].min(axis=1)
max_x = car.iloc[:, [1,2,3,4]].max(axis=1)
min_y = car.iloc[:, [5,6,7,8]].min(axis=1)
max_y = car.iloc[:, [5,6,7,8]].max(axis=1)

table = pd.DataFrame(np.array([car.iloc[:, 0], min_x, min_y, max_x, max_y, car.iloc[:, -1]]).T)
#table.columns = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
#table.to_csv('annotation_all.txt', sep=' ', index=None)

table.iloc[:, [1,2,3,4]] = table.iloc[:, [1,2,3,4]].clip(1, 511)
for i in range(len(table)):
    row = table.iloc[i, [0, -1, 1, 2, 3, 4]].to_list()
    f = open(data_path + 'labels/' + row[0][:-4] + '.txt', 'a')
    for j in range(1, 6):
        f.write(str(row[j]) + ' ')
    f.write('\n')
    f.close()

    image = glob.glob(os.path.join(data_path, '*', '*', row[0][:-4] + '.png'))
    if len(image)>0:
        new_path = shutil.move(image[0], os.path.join(data_path, 'img'))