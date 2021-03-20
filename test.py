import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.transform import resize
from sklearn.decomposition import PCA
import pickle
from skimage.io import imread
import os

for file in os.listdir("./test/test-1")[:100]:
    print(file)
    imga = imread("./test/test-1/%s" % file)
    imgs = cv.cvtColor(imga, cv.COLOR_RGB2BGR)

