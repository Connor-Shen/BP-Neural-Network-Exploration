import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
import cv2

dic = loadmat("D:/desktop/NN&DL/project_1_release/codes/digits.mat")
idx = 5
img = dic["X"][idx, :].reshape(16,16)
flip_21 = np.fliplr(img) # 水平翻转
flip_22 = np.transpose(img)	# 转置

M = cv2.getRotationMatrix2D((8,8), 270, 1.0)
rotation_11 = cv2.warpAffine(img, M, (16,16)) # 旋转

resize_11 = np.resize(img, (24,24)) # 改变大小

dx, dy = 5, 5  # dx 向右偏移量, dy 向下偏移量
MAT = np.float32([[1, 0, dx], [0, 1, dy]])  # 构造平移变换矩阵
# dst = cv2.warpAffine(img, MAT, (cols, rows))  # 默认为黑色填充
dst = cv2.warpAffine(img, MAT, (16, 16), borderValue=(255,255,255))  # 设置白色填充

tr_data = dic["X"]
tr_data_rotation = []
for idx, data in enumerate(tr_data):
    img = data.reshape(16,16)
    if 0<= idx <= 2000:
        M = cv2.getRotationMatrix2D((8, 8), 270, 1.0)
        rotation_img = cv2.warpAffine(img, M, (16, 16))
        tr_data_rotation.append(rotation_img)
    if 2000< idx <= 4000:
        M = cv2.getRotationMatrix2D((8, 8), 90, 1.0)
        rotation_img = cv2.warpAffine(img, M, (16, 16))
        tr_data_rotation.append(rotation_img)
    if 4000< idx <= 5000:
        M = cv2.getRotationMatrix2D((8, 8), 180, 1.0)
        rotation_img = cv2.warpAffine(img, M, (16, 16))
        tr_data_rotation.append(rotation_img)

img_new = tr_data_rotation[idx]
print(dic["y"][idx])
print(len(tr_data_rotation))
plt.imshow(img,cmap="gray")
plt.show()
plt.imshow(img_new,cmap="gray")
plt.show()