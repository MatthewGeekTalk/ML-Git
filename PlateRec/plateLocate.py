import cv2
import numpy as np
import matplotlib.pyplot as plt

class CPlateLocate:

    def __init__(self):
        self.m_GaussianBlurSize = 0.
        self.img = object
        self.imgOrg = object
        self.im2 = object
        self.morphH = 0
        self.morphW = 0
        self.region = []
        self.plates = []

    def read_img(self, path):
        self.img = cv2.imread(path)
        self.imgOrg = cv2.imread(path)

    def plate_locate(self):
        self.img = self.__gaussian_blur()
        self.img = self.__img_gray()
        self.img = self.__img_sobel()
        self.img = self.__img_binary()
        self.img = self.__img_morph_close()
        cv2.imshow('morph', self.img)
        self.region = self.__findPlateNumberRegion()
        self.plates = self.__detectRegion()

    def set_gaussian_size(self, gaussian_blur_size):
        self.m_GaussianBlurSize = gaussian_blur_size

    def set_morph_hw(self, morph_w, morph_h):
        self.morphW = morph_w
        self.morphH = morph_h

    def __gaussian_blur(self):
        return cv2.GaussianBlur(self.img, (self.m_GaussianBlurSize, self.m_GaussianBlurSize), 0, 0, cv2.BORDER_DEFAULT)

    def __img_gray(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def __img_sobel(self):
        return cv2.Sobel(self.img, cv2.CV_8U, 1, 0, ksize=3)

    def __img_binary(self):
        ret, binary = cv2.threshold(self.img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        return binary

    def __img_morph_close(self):
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphW, self.morphH))
        return cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, element)

    def __findPlateNumberRegion(self):
        region = []
        img_find = self.img.copy()
        im2, contours, hierarchy = cv2.findContours(img_find, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area < 2000):
                continue
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])
            ratio = float(width) / float(height)
            if (ratio > 5 or ratio < 2):
                continue
            region.append(box)
        return region

    def __detectRegion(self):
        plates = []
        for box in self.region:
            cv2.drawContours(self.imgOrg, [box], 0, (0, 255, 0), 2)
            ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs)

            x1 = box[xs_sorted_index[0], 0]
            x2 = box[xs_sorted_index[3], 0]

            y1 = box[ys_sorted_index[0], 1]
            y2 = box[ys_sorted_index[3], 1]
            img_org2 = self.imgOrg.copy()
            img_plate = img_org2[y1:y2, x1:x2]
            plates.append(img_plate)
        return plates

    def img_show(self):
        cv2.imshow('img', self.imgOrg)
        for i in range(len(self.plates)):
            cv2.imshow('plates_' + str(i), self.plates[i] )
        cv2.waitKey(0)

if __name__ == '__main__':
    path = input('Please input your image path:')
    plate_locate = CPlateLocate()
    plate_locate.read_img(path)
    plate_locate.set_gaussian_size(5)
    plate_locate.set_morph_hw(17, 3)
    plate_locate.plate_locate()
    plate_locate.img_show()

# C:\\Users\\zzfancitizen\\Desktop\\plate4.jpg