import cv2
import numpy as np
import math
import os

class SobelPlateLocate:
    def __init__(self):
        self.rect = []
    def charsSegment(self,img,color):
        img_gay = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_threshold = self.spatial_ostu(img_gay,color)
        im2, contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    print('123 ')