import cv2
import numpy as np
import math
import os

class charsSegment:
    def __init__(self):
        self.rect = []
        self.img = object

    def read_img(self, img_path):
        self.img = cv2.imread(img_path)
        return self.img

    def charsSegment(self,img,color):
        img_gay = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_threshold = self.spatial_ostu(img_gay,color)
        im2, contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area <= 50:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            print(x,y,w,h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
            cv2.imshow('plates_' + str(i), img)
        cv2.waitKey(0)
    def spatial_ostu(self,src,color):
        if color == 'BLUE':
            ret,img_thre = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        elif color == 'YELLOW':
            ret, img_thre = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        elif color == 'WHITE':
            ret, img_thre = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        else:
            ret, img_thre = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        return img_thre


if __name__ == '__main__':
    charsSegment = charsSegment()
    img = charsSegment.read_img('000_1.jpg')
    charsSegment.charsSegment(img,'BLUE')