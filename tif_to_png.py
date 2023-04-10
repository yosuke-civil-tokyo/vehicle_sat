import cv2
import os
import argparse

os.makedirs('vott_folder', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-img', type=str)
parser.add_argument('-vott', type=str)
args = parser.parse_args()

img_tif = cv2.imread(args.img)

x_len = img_tif.shape[0]
y_len = img_tif.shape[1]

# split image into 512x512 .png images
num = 1
for i in range(x_len//512):
    for j in range(y_len//512):
        img_png = img_tif[i*512:(i+1)*512, j*512:(j+1)*512, :]
        cv2.imwrite(os.path.join(args.vott, '{}.png'.format(num)), img_png)
        num += 1

    print('Now {}'.format(num))