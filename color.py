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


# ans = ['长颈鹿', '火车', '车', '斑马', '狐狸', '飞机', '狗', '鲸鱼', '羊', '熊', '猴子', '雕像',
#        '牛', '老虎', '狮子', '塔', '马', '麋鹿', '考拉', '鸟', '玫瑰花', '菊花', '鸡冠花',
#        '康乃馨花', '红掌花', '郁金香花', '鸢尾花', '荷花', '百合花', '梨花', '牡丹花', '紫罗兰花', '一串红花',
#        '牵牛花', '玲花', '栀子花', '茉莉花', '杜鹃花', '桂花', '马蹄莲花']
#
class color:
    X = []
    Y = []


co1 = color()
co2 = color()

co = color()
# 提取特征向量
for i in range(1, 41):
    for f in os.listdir("./test2020/%s/" % i):
        # 颜色直方图
        Images = cv.imread("./test2020/%s/%s" % (i, f))
        image = cv.resize(Images, (256, 256), interpolation=cv.INTER_CUBIC)
        # print(image.dtype)
        hist_bgr = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0.0, 255.0, 0.0, 255.0, 0.0, 255.0])
        f1 = hist_bgr.flatten()
        co.X.append(f1)
        co.Y.append(i)

# X是特征向量集、y是物品类别集
co.X = np.array(co.X)
co.Y = np.array(co.Y)

with open("color_test.pkl", "wb") as f:
    pickle.dump(co, f)
with open("color_train.pkl", "wb") as f:
    pickle.dump(co, f)

# 测试过程
with open("color_train.pkl", "rb") as f:
    co1 = pickle.load(f)
with open("color_test.pkl", "rb") as f:
    co2 = pickle.load(f)

svc = RandomForestClassifier(n_estimators=100,
                             random_state=50,
                             max_features='sqrt',
                             n_jobs=-1, verbose=1)
# svc = SVC()
# svc = KNeighborsClassifier()
svc.fit(co1.X, co1.Y)
with open("color.pkl", "wb") as f:
    pickle.dump(svc, f)
re = svc.predict(co2.X)
print(confusion_matrix(co2.Y, re))
print(classification_report(co2.Y, re))
