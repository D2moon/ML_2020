import numpy as np
import pickle
import os
import cv2 as cv
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report


class SOLVE:
    X = []
    Y = []
    lens = []


f = open("sift2.pkl", "rb")
sift = pickle.load(f)
f.close()
f = open("sift_test.pkl", "rb")
test = pickle.load(f)
f.close()
# f = open("kmeans.pkl", "rb")
# ks = pickle.load(f)
# f.close()

svc = SVC()
# svc = RandomForestClassifier(n_estimators=100,
#                              random_state=50,
#                              max_features='sqrt',
#                              n_jobs=-1, verbose=1)

# svc = KNeighborsClassifier()
# svc.fit(sift.X, sift.Y)
with open("sift.pkl", "rb") as f:
    svc = pickle.load(f)

# test = SOLVE()
# path = "./test2020"
# for i in range(1, 41):
#     for f in os.listdir(path + "/%s" % i):
#         img = cv.imread(path + "/%s/%s" % (i, f))
#         img = cv.resize(img, (256, 256))
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.1)
#         kp = sift.detect(gray, None)
#         kp, des = sift.compute(gray, kp)
#         if np.shape(kp)[0] < 10:
#             print(i, f)
#             os.remove(path + "/%s/%s" % (i, f))
#             continue
#         # print(i, f, np.shape(kp)[0])
#         test.Y.append(i)
#         test.lens.append(np.shape(kp)[0])
#         now = np.zeros(80)
#         re = ks.predict(des)
#         for t1 in re:
#             now[t1] = now[t1] + 1
#         test.X.append(now)
#
# test.Y = np.array(test.Y)
# test.X = np.array(test.X)

# with open("sift_test.pkl", "wb") as f:
#     pickle.dump(test, f)
#
re = svc.predict(test.X)
print(confusion_matrix(test.Y, re))
print(classification_report(test.Y, re))
