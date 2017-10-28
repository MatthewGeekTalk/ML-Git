import sys
import os
import numpy
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('./'))
from sobelPlateLocate import SobelPlateLocate
from plate_validate import PlateValidate
from charsSegment import charsSegment

IS_PLATE = [0, 1]


class PlateRec(object):
    def __init__(self):
        self.path = ""
        self.sobel_plates = []

    def main(self):
        self.sobel_plates = self.__get_img()

    def __get_img(self):
        plate_sobel = SobelPlateLocate()
        plate_sobel.read_img(self.path)
        plate_sobel.set_gaussian_size(5)
        plate_sobel.set_morph_hw(17, 3)
        plate_sobel.set_verify_value(1, 100, 4, .5)
        plate_sobel.plate_locate()
        return plate_sobel.return_plates()


if __name__ == '__main__':
    sobel_plates = []
    img_plate = []
    region_plate = []
    bool_while = True

    print('Image path: %s' % str(os.path.abspath('../Material')).replace('\\', '\\\\'))
    path = input('Please input your image path:')

    img = cv2.imread(path, cv2.COLOR_BGR2RGB)

    plate_sobel = SobelPlateLocate()
    plate_sobel.read_img(path)
    plate_sobel.set_size(20, 72)
    plate_sobel.set_gaussian_size(5)
    plate_sobel.set_morph_hw(17, 3)
    plate_sobel.set_verify_value(1, 100, 4, .5)
    plate_sobel.plate_locate()
    sobel_plates = plate_sobel.return_plates()
    sobel_regions = plate_sobel.return_regions()

    plate_validate = PlateValidate(sobel_plates)
    imgs, labels = plate_validate.main()

    for i in range(len(imgs)):
        if labels[i] == IS_PLATE:
            img_plate.append(imgs[i])
            region_plate.append(sobel_regions[i])

            plt.axis('off')
            plt.imshow(imgs[i])
            plt.show()
            print(labels[i])

    for i in range(len(region_plate)):
        img = cv2.drawContours(img, [region_plate[i]], 0, (0, 255, 0), 2)

    # plt.imshow(img)
    # plt.show()

    char_detect = charsSegment()
    for img in img_plate:
        char_img = char_detect.read_img(img)
        chars = char_detect.charsSegment(char_img, 'BLUE')

        for char in chars:
            plt.axis('off')
            plt.imshow(char)
            plt.show()
