import numpy as np
import tornado.ioloop
import tornado.web
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
from PIL import Image
from io import BytesIO
import cv2 as cv
import pickle
from sklearn.svm import SVC
import os

LBP_radius = 4
LBP_points = 25


class color:
    X = []
    Y = []


class SOLVE:
    X = []
    Y = []
    lens = []


color_p = [0.35, 0.38, 0.51, 0.49, 0.46, 0.62, 0.43, 0.75, 0.43, 0.39,
           0.67, 0.37, 0.38, 0.38, 0.34, 0.37, 0.27, 0.55, 0.41, 0.57,
           0.44, 0.65, 0.64, 0.55, 0.59, 0.68, 0.65, 0.62, 0.50, 0.60,
           0.57, 0.66, 0.72, 0.66, 0.58, 0.61, 0.48, 0.54, 0.66, 0.54]
sift_p = [0.61, 0.35, 0.44, 0.67, 0.11, 0.20, 0.20, 0.25, 0.27, 0.22,
          0.10, 0.15, 0.33, 0.45, 0.15, 0.22, 0.28, 0.33, 0.21, 0.18,
          0.21, 0.42, 0.21, 0.37, 0.40, 0.14, 0.18, 0.25, 0.38, 0.49,
          0.21, 0.31, 0.32, 0.13, 0.36, 0.46, 0.46, 0.17, 0.50, 0.35]
lbp_p = [0.17, 0.30, 0.43, 0.58, 0.16, 0.47, 0.19, 0.46, 0.32, 0.20,
         0.60, 0.50, 0.53, 0.24, 0.31, 0.32, 0.12, 0.37, 0.20, 0.35,
         0.24, 0.19, 0.13, 0.39, 0.33, 0.38, 0.35, 0.24, 0.27, 0.45,
         0.21, 0.29, 0.29, 0.21, 0.94, 0.58, 0.15, 0.00, 0.56, 0.26]
hog_p = [0.00, 0.67, 0.77, 0.43, 0.25, 0.73, 0.31, 0.53, 0.33, 0.29,
         0.67, 0.47, 0.58, 0.27, 0.46, 0.58, 0.40, 0.30, 0.35, 0.37,
         0.60, 0.27, 0.39, 0.50, 0.29, 0.43, 0.38, 0.27, 0.31, 0.00,
         0.26, 0.22, 0.26, 0.27, 0.63, 0.67, 0.62, 0.43, 0.62, 0.27]

f = open("sift.pkl", "rb")
sift_svc = pickle.load(f)
f.close()
f = open("kmeans.pkl", "rb")
ks = pickle.load(f)
f.close()

color_for = RandomForestClassifier(n_estimators=100,
                             random_state=50,
                             max_features='sqrt',
                             n_jobs=-1, verbose=1)
with open("color.pkl", "rb") as f:
    color_for = pickle.load(f)

lbp_for = RandomForestClassifier(n_estimators=100,
                             random_state=50,
                             max_features='sqrt',
                             n_jobs=-1, verbose=1)
with open("lbp.pkl", "rb") as f:
    lbp_for = pickle.load(f)

hog_svc = SVC()
with open("hog.pkl", "rb") as f:
    hog_svc = pickle.load(f)

re = []
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        title = "NEU_MICSTU_GROUP6"
        # path = r"/home/micstu/images/g6/11/"
        # imgs = os.listdir(path)
        self.render("webserver1.html", title = title)

class Images(tornado.web.RequestHandler):
    def get(self):
	    self.render("webserver2.html")

class Monkey(tornado.web.RequestHandler):
    def get(self):
        path = r"/home/micstu/static/g6/11/train/"
        imgs = os.listdir(path)
        imgs = imgs[:50]
        for t in imgs:
            t = "/static/g6/11/static/"+t
        self.render("monkey.html", imgs= imgs)

class Peony(tornado.web.RequestHandler):
    def get(self):
        path = r"/home/micstu/static/g6/2111/train/"
        imgs = os.listdir(path)
        imgs = imgs[:50]
        for t in imgs:
            t = "/static/g6/2111/static/"+t
        self.render("peony.html", imgs= imgs)

class Index(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", re = re)

class Out(tornado.web.RequestHandler):
    def get(self):
        self.render("out.html", re= re)

if __name__ == "__main__":
    for i in range(1):
        X = []
        names = []
        sift_test = []
        color_test = []
        lbp_test = []
        hog_test = []
        for file in os.listdir("./exam-final"):
            names.append(file)
            imga = imread("./exam-final/%s" % file)
            imgs = cv.cvtColor(imga, cv.COLOR_RGB2BGR)

            # sift
            img = cv.resize(imgs, (256, 256))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.1)
            kp = sift.detect(gray, None)
            kp, des = sift.compute(gray, kp)
            if np.shape(kp)[0] < 10:
                print(file)
                continue
            # print(i, f, np.shape(kp)[0])
            now = np.zeros(80)
            re = ks.predict(des)
            for t1 in re:
                now[t1] = now[t1] + 1
            sift_test.append(now)

            # color
            image = cv.resize(imgs, (256, 256), interpolation=cv.INTER_CUBIC)
            hist_bgr = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0.0, 255.0, 0.0, 255.0, 0.0, 255.0])
            f1 = hist_bgr.flatten()
            color_test.append(f1)

            # lbp
            image = cv.resize(imgs, (256, 256), interpolation=cv.INTER_CUBIC)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            LBP = local_binary_pattern(gray, LBP_points, LBP_radius, 'default')
            LBP = LBP.astype(np.uint8)
            hist_grey1 = cv.calcHist([LBP], [0], None, [128], [0.0, 255.0])
            f3 = hist_grey1.flatten()
            lbp_test.append(f3)

            # hog
            imga = imread("./exam-final/%s" % file)
            img = resize(imga, (128, 64))
            fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)
            hog_test.append(fd)

        sift_test = np.array(sift_test)
        sift_re = sift_svc.predict(sift_test)

        color_test = np.array(color_test)
        color_re = color_for.predict(color_test)

        lbp_test = np.array(lbp_test)
        lbp_re = lbp_for.predict(lbp_test)

        hog_test = np.array(hog_test)
        hog_re = hog_svc.predict(hog_test)

        ans = []
        final = []
        for p1 in range(0, 100):
            pt = []
            for p2 in range(0, 40):
                pt.append(0)
            ans.append(pt)
        nt = 0
        for nt in range(0, 100):
            tt1 = sift_re[nt]
            tt2 = lbp_re[nt]
            tt3 = hog_re[nt]
            tt4 = color_re[nt]
            ans[nt][tt1 - 1] = ans[nt][tt1 - 1] + sift_p[tt1 - 1]
            ans[nt][tt2 - 1] = ans[nt][tt2 - 1] + lbp_p[tt2 - 1]
            ans[nt][tt3 - 1] = ans[nt][tt3 - 1] + hog_p[tt3 - 1]
            ans[nt][tt4 - 1] = ans[nt][tt4 - 1] + color_p[tt4 - 1]
            ind = ans[nt].index(max(ans[nt])) + 1
            final.append(ind)
        for i in range(len(final)):
            if final[i] > 20:
                final[i] = final[i] + 2080
        with open("result.txt", "w") as f:
            for i in range(100):
                f.write(names[i] + ", " + str(final[i]))
                f.write('\n')
        print(names)

settings = {
    "debug": True,
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
    }

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/images", Images),
        (r"/monkey", Monkey),
        (r"/peony", Peony),
        (r"/index", Index),
        (r"/upload", Upload),
        (r"/out", Out),
        ], **settings)


print("server starting..")

# app = make_app()
# app.listen(8888)
# tornado.ioloop.IOLoop.current().start()