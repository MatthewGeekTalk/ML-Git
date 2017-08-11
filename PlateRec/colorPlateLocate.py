import cv2
import numpy
import os


class ColorPlateLocate:
    def __init__(self):
        self.max_sv = 0
        self.minref_sv = 0
        self.minabs_sv = 0

        self.min_blue = 0
        self.max_blue = 0
        self.min_yellow = 0
        self.max_yellow = 0

        self.img = object
        self.imgOrg = object
        self.hsv = object

    def read_img(self, img_path):
        self.img = cv2.imread(img_path)
        self.imgOrg = self.img.copy()

    def set_img_hsv(self, max_sv, minref_sv, minabs_sv, min_blue, max_blue, min_yellow, max_yellow):
        self.max_sv = max_sv
        self.minref_sv = minref_sv
        self.minabs_sv = minabs_sv

        self.min_blue = min_blue
        self.max_blue = max_blue
        self.min_yellow = min_yellow
        self.max_yellow = max_yellow

    def plate_locate(self):
        pass

    def __set_bgr2hsv(self):
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

if __name__ == "__init__":
    print("Image path: %s" % str(os.path.abspath('../Material')).replace('\\', '\\\\'))
    path = input("Please input your image path:")
    plate_locate = ColorPlateLocate()
    plate_locate.read_img(path)
    plate_locate.set_img_hsv(255, 64, 95, 100, 140, 15, 40)
    plate_locate.plate_locate()

