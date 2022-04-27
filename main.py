from omegaconf import OmegaConf
import os
import numpy as np
from PIL import Image
import math


def getColorList(path: str):
    config = OmegaConf.load(path)
    number = list(config["number"])
    value = list(config["value"])
    le = len(value)
    outList = np.ones([2, 4, le], dtype=int)
    outList2 = np.ones([le, 3], dtype=int)
    for i in range(le):
        outList[1, 0, i] = -1 * int(value[i][0:2], 16) / 2
        outList[1, 1, i] = -1 * int(value[i][0:2], 16)
        outList[1, 2, i] = -1 * int(value[i][2:4], 16)
        outList[1, 3, i] = -1 * int(value[i][4:6], 16)
        outList2[i, 0] = int(value[i][0:2], 16)
        outList2[i, 1] = int(value[i][2:4], 16)
        outList2[i, 2] = int(value[i][4:6], 16)
    return number, outList, outList2


def getImage(path: str):
    ori = Image.open(path).convert("RGB")
    ori = np.array(ori)
    image = ori.reshape(-1, 3)
    he, we, _ = ori.shape
    tempImage = np.ones([he*we, 4, 2], dtype='uint8')
    tempImage[:, 0, 0] = image[:, 0] / 2
    tempImage[:, 1, 0] = image[:, 0]
    tempImage[:, 2, 0] = image[:, 1]
    tempImage[:, 3, 0] = image[:, 2]
    return he, we, tempImage


relpath = os.path.dirname(os.path.abspath(__file__))
configPath = os.path.join(relpath, r"configs\colorList.yaml")
imgPath = os.path.join(relpath, r"src\lena.png")
colorNumber, valueList, RGBList = getColorList(configPath)
h, w, img = getImage(imgPath)

# algorithm
indices = np.ones([h*w, 1], dtype=int)*(-1)
for i in range(h*w):
    temp = np.zeros([4, 4], dtype=float)
    max_c = 1e5
    for j in range(len(colorNumber)):
        temp[:, :] = np.dot(img[i, :, :], valueList[:, :, j])
        delta_c = math.sqrt((2 + temp[0, 0] / 256) * math.pow(temp[1, 1], 2) + 4 * math.pow(temp[2, 2], 2) +
                            (2 + (255 - temp[0, 0]) / 256) * math.pow(temp[3, 3], 2))
        if delta_c <= max_c:
            max_c = delta_c
            indices[i] = j
        temp = temp * 0

rec = np.zeros([h*w, 3], dtype='uint8')
for i in range(h*w):
    rec[i, :] = RGBList[indices[i], :]
rec = rec.reshape(h, w, -1)

recImgPath = os.path.join(relpath, r"src\lena_rec.jpg")
Image.fromarray(rec).save(recImgPath)
