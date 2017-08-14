import cv2
import numpy as np

img = object
img = cv2.imread('C:\\Users\\i072179\\PycharmProjects\\ML-Git\\Material\\plate1.jpg')
imgOrg = img.copy()
img = cv2.GaussianBlur(img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)
im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[4]
print(cnt)

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(im2, contours, 0, (0, 0, 255), 3)

cv2.imshow('im2', im2)
cv2.waitKey(0)

