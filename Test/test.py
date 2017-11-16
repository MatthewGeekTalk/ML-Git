import cv2
import numpy as np

img = object
img = cv2.imread('00.jpg')
imgOrg = img.copy()
img_gay = cv2.cvtColor(imgOrg, cv2.COLOR_BGR2GRAY)
BLUE_MIN = np.array([0, 0, 0], np.uint8)
BLUE_MAX = np.array([255, 255, 50], np.uint8)

dst = cv2.inRange(img, BLUE_MIN, BLUE_MAX)
print(img_gay.shape, dst.shape)
z = cv2.countNonZero(dst)
print(z)
# ret,img_thre = cv2.threshold(img_gay,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)
# cv2.imshow('im2', img_thre)
# cv2.waitKey(0)
