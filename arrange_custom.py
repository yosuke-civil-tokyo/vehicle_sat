import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-custom', type=str)
args = parser.parse_args()

target_dataset = args.custom
with open(os.path.join(target_dataset, 'train_true.txt'), 'w') as f:
    f.write('')

labels = os.listdir(path=target_dataset + 'labels')

# remove images including no vehicle in them
for text in labels:
    length = len(open(os.path.join(target_dataset, 'labels/', text)).read())
    if length != 0:
        with open(os.path.join(target_dataset, 'train_true.txt'), mode='a') as f:
            f.write(text[:-3]+'png\n')
    else:
        os.remove(os.path.join(target_dataset, 'images/', text[:-3] + 'png'))
        os.remove(os.path.join(target_dataset, 'labels/', text))