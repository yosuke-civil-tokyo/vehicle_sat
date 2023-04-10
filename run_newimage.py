import sys
from pathlib import Path

import cv2
#from IPython.display import Image, display

# 以下の3つのパスは適宜変更してください
yolov3_path = Path("./")  # git clone した pytorch_yolov3 ディレクトリのパスを指定してください。
config_path = yolov3_path / "config/yolov3_custom.yaml"  # yolov3_custom.yaml のパスを指定してください
weights_path = yolov3_path / "train_output/yolov3_final.pth"  # 重みのパスを指定してください

sys.path.append(str(yolov3_path))
from yolov3.detector import Detector
"""
def imshow(img):
    #ndarray 配列をインラインで Notebook 上に表示する。
    ret, encoded = cv2.imencode(".jpg", img)
    display(Image(encoded))
"""

# 検出器を作成する。
detector = Detector(config_path, weights_path)

# 画像を読み込む。
args = sys.argv
img = cv2.imread('custom_dataset/images/' + str(args[1]))

# 検出する。
detection = detector.detect_from_imgs(img)[0]

# 検出結果を画像に描画する。
for bbox in detection:
    cv2.rectangle(
        img,
        (int(bbox["x1"])-1, int(bbox["y1"])-1),
        (int(bbox["x2"])+1, int(bbox["y2"])+1),
        color=(0, 255, 0),
        thickness=1,
    )

cv2.imwrite('output/' + str(args[1]), img)
