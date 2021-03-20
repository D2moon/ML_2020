import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage.feature import local_binary_pattern
import cv2 as cv
import pickle
from sklearn.svm import SVC
import os

LBP_radius = 4
LBP_points = 25


class color:
    X = []
    Y = []


# co = color()
# # 提取特征向量
# for i in range(1, 41):
#     for f in os.listdir("./train2020/%s/" % i):
#         # 颜色直方图
#         Images = cv.imread("./train2020/%s/%s" % (i, f))
#         image = cv.resize(Images, (256, 256), interpolation=cv.INTER_CUBIC)
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#         LBP = local_binary_pattern(gray, LBP_points, LBP_radius, 'default')
#         LBP = LBP.astype(np.uint8)
#         hist_grey1 = cv.calcHist([LBP], [0], None, [128], [0.0, 255.0])
#         f3 = hist_grey1.flatten()
#         co.X.append(f3)
#         co.Y.append(i)
#         # print("f3:", np.mean(f3))
#
# # X是特征向量集、y是物品类别集
# co.X = np.array(co.X)
# co.Y = np.array(co.Y)

# with open("lbp_test.pkl", "wb") as f:
#     pickle.dump(co, f)
# with open("lbp_train.pkl", "wb") as f:
#     pickle.dump(co, f)

# 测试过程
with open("lbp_train.pkl", "rb") as f:
    co1 = pickle.load(f)
with open("lbp_test.pkl", "rb") as f:
    co2 = pickle.load(f)

svc = RandomForestClassifier(n_estimators=100,
                             random_state=50,
                             max_features='sqrt',
                             n_jobs=-1, verbose=1)
# svc = SVC()
# svc = KNeighborsClassifier()
svc.fit(co1.X, co1.Y)
with open("lbp.pkl", "wb") as f:
    pickle.dump(svc, f)
re = svc.predict(co2.X)
print(confusion_matrix(co2.Y, re))
print(classification_report(co2.Y, re))
