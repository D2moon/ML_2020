import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
import cv2 as cv
import pickle
from sklearn.svm import SVC
import os


class color:
    X = []
    Y = []


co = color()
# 提取特征向量
for i in range(1, 41):
    for f in os.listdir("./train2020/%s/" % i):
        # print(i, f)
        img = imread("./train2020/%s/%s" % (i, f))
        img = resize(img, (128, 64))

        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)
        co.X.append(fd)
        co.Y.append(i)

# X是特征向量集、y是物品类别集
co.X = np.array(co.X)
print(len(co.X[0]))
co.Y = np.array(co.Y)
with open("hog_train.pkl", "wb") as f:
    pickle.dump(co, f)

# co1 = color()
# # 提取特征向量
# for i in range(2, 41):
#     for f in os.listdir("./test2020/%s/" % i):
#         img = imread("./test2020/%s/%s" % (i, f))
#         img = resize(img, (128, 64))
#
#         fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
#                             cells_per_block=(2, 2), visualize=True)
#         co1.X.append(fd)
#         co1.Y.append(i)
#
# # X是特征向量集、y是物品类别集
# co1.X = np.array(co1.X)
# co1.Y = np.array(co1.Y)

# with open("hog_test.pkl", "wb") as f:
#     pickle.dump(co1, f)

# 测试过程
with open("hog_train.pkl", "rb") as f:
    co1 = pickle.load(f)
with open("hog_test.pkl", "rb") as f:
    co2 = pickle.load(f)

print(co2.X.shape)
# svc = RandomForestClassifier(n_estimators=100,
#                              random_state=50,
#                              max_features='sqrt',
#                              n_jobs=-1, verbose=1)
# svc = SVC()
# svc = KNeighborsClassifier()
# svc.fit(co1.X, co1.Y)
# print(co1.X)
print("hello")
with open("hog.pkl", "rb") as f:
    svc = pickle.load(f)
a = co2.X
b = co2.Y
print(a.shape)
re = svc.predict(a)
print(confusion_matrix(b, re))
print(classification_report(b, re))
