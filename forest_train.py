import datetime
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage.feature import local_binary_pattern

import cv2 as cv
import pickle
import os

# GLCM参数
GLCM_dis = [1, 3]
GLCM_ang = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_level = 256

# LBP参数
LBP_radius = 2
LBP_points = 10

# ans = ['长颈鹿', '火车', '车', '斑马', '狐狸', '飞机', '狗', '鲸鱼', '羊', '熊', '猴子', '雕像',
#        '牛', '老虎', '狮子', '塔', '马', '麋鹿', '考拉', '鸟', '玫瑰花', '菊花', '鸡冠花',
#        '康乃馨花', '红掌花', '郁金香花', '鸢尾花', '荷花', '百合花', '梨花', '牡丹花', '紫罗兰花', '一串红花',
#        '牵牛花', '玲花', '栀子花', '茉莉花', '杜鹃花', '桂花', '马蹄莲花']
#
X = []
Y = []

# 提取特征向量
for i in range(1, 41):
    for f in os.listdir("./train2020/%s/" % i):
        # 颜色直方图
        Images = cv.imread("./train2020/%s/%s" % (i, f))
        image = cv.resize(Images, (200, 180), interpolation=cv.INTER_CUBIC)
        # print(image.dtype)
        hist_bgr = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0.0, 255.0, 0.0, 255.0, 0.0, 255.0])
        print(hist_bgr)
        f1 = hist_bgr.flatten()
        # print("f1:", np.mean(f1))

        # GLCM
        grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        GLCM = greycomatrix(grey, GLCM_dis, GLCM_ang, GLCM_level)
        f2 = []
        for prop in {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}:
            temp = greycoprops(GLCM, prop)
            f2.append(np.mean(temp))
        # print("f2:", np.mean(f2))
        f1 = np.append(f1, f2)

        # LBP
        LBP = local_binary_pattern(grey, LBP_points, LBP_radius, 'default')
        LBP = LBP.astype(np.uint8)
        # print(LBP.dtype)
        hist_grey1 = cv.calcHist([LBP], [0], None, [8], [0.0, 255.0])
        f3 = hist_grey1.flatten()
        # print("f3:", np.mean(f3))
        f1 = np.append(f1, f3)

        X.append(f1)
        Y.append(i)

# X是特征向量集、y是物品类别集
X = np.array(X)
Y = np.array(Y)

# 训练过程
# knn = KNeighborsClassifier(n_neighbors=k)
# knn.fit(X_train, y_train)
# tree1 = tree.DecisionTreeClassifier()
# tree1.fit(X_train, y_train)
model = RandomForestClassifier(n_estimators=100,
                               random_state=50,
                               max_features='sqrt',
                               n_jobs=-1, verbose=1)
model.fit(X, Y)

with open("FOREST.pkl", "wb") as f:
    pickle.dump(model, f)


# 测试过程
# re = knn.predict(X_test)
#
# print(confusion_matrix(y_test, re))
# print(classification_report(y_test, re))
#
# end_time = datetime.datetime.now()
# print(end_time - start_time)






