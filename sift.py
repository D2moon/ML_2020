import numpy as np
import cv2 as cv
import pickle
import os
from sklearn.cluster import KMeans


# import cv2.cv2 as cv

# part1
class SIFT:
    ids = []
    lens = []
    data = []


#
#
# sfs = SIFT()
# path = "./train2020"
# for i in range(1, 41):
#     for f in os.listdir(path+"/%s" % i):
#         img = cv.imread(path+"/%s/%s" % (i, f))
#         img = cv.resize(img, (256, 256))
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.1)
#         kp = sift.detect(gray, None)
#         kp, des = sift.compute(gray, kp)
#         if np.shape(kp)[0] < 10:
#             print(i, f)
#             continue
#         print(i, f, np.shape(kp)[0])
#         sfs.ids.append(i)
#         sfs.lens.append(np.shape(kp)[0])
#         re = des.tolist()
#         sfs.data = sfs.data+re
#
# sfs.data = np.array(sfs.data)
# with open("sift.pkl", "wb") as f:
#     pickle.dump(sfs, f)
# print("特征提取完成")
#
#
# ks = KMeans(n_clusters=80)
# ks.fit(sfs.data)
#
# with open("kmeans.pkl", "wb") as f:
#     pickle.dump(ks, f)
# print("聚类完成")


# part2
class SOLVE:
    X = []
    y = []
    lens = []


sfs = SIFT()
path = "./train2020"
for i in range(1, 41):
    for f in os.listdir(path+"/%s" % i):
        img = cv.imread(path+"/%s/%s" % (i, f))
        img = cv.resize(img, (256, 256))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.1)
        kp = sift.detect(gray, None)
        kp, des = sift.compute(gray, kp)
        if np.shape(kp)[0] < 10:
            print(i, f)
            os.remove(path+"/%s/%s" % (i, f))
            continue
        # print(i, f, np.shape(kp)[0])
        sfs.ids.append(i)
        sfs.lens.append(np.shape(kp)[0])

with open("sift1.pkl", "wb") as f:
    pickle.dump(sfs, f)
print("特征提取完成")

# re = SOLVE()
# f = open("kmeans.pkl", "rb")
# ks = pickle.load(f)
# print(ks.labels_)
# f.close()
#
# re.Y = sfs.ids
# re.lens = sfs.lens
#
#
# t1 = 0
# t2 = []
# for num in sfs.lens:
#     now = np.zeros(80)
#     for i in range(num):
#         now[ks.labels_[t1]] = now[ks.labels_[t1]] + 1
#         t1 = t1 + 1
#     print(now)
#     t2.append(now)
# re.X = np.array(t2)
# with open("sift2.pkl", "wb") as f:
#     pickle.dump(re, f)
