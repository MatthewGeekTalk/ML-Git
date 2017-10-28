import sys
import os
import numpy
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('./'))
from sobelPlateLocate import SobelPlateLocate
from colorPlateLocate import ColorPlateLocate
from plate_validate import PlateValidate
from charsSegment import charsSegment

IS_PLATE = [0, 1]


class PlateRec(object):
    def __init__(self):
        self.path = ""
        self.sobel_plates = []

        self._img = object
        self._img_con_sobel = object
        self._img_con_color = object
        self._plates_sobel = object
        self._regions_sobel = object
        self._plates_color = object
        self._regions_color = object

        self.plate_validate = PlateValidate()

        self._plate_with_no = []

    def main(self):
        self._plates_sobel, self._regions_sobel = self.__detect_plate_sobel()
        # self._plates_color, self._regions_color = self.__detect_plate_color()
        self._img_con_sobel = self.__prepare_contours_img(regions=self._regions_sobel)
        # self._img_con_color = self.__prepare_contours_img(regions=self._regions_color)

        for i in range(len(self._plates_sobel)):
            chars = self.__detect_char(plate=self._plates_sobel[i])
            self._plate_with_no.append({'id': i, 'key': chars})

    def __prepare_contours_img(self, regions):
        ori_img = self._img.copy()
        return self.__draw_contours(img=ori_img, plate_regions=regions)

    @staticmethod
    def __detect_char(plate):
        char_detect = charsSegment()
        char_img = char_detect.read_img(plate)
        return char_detect.charsSegment(char_img, 'BLUE')

    def __detect_plate_sobel(self):
        img_plate = []
        region_plate = []

        plate_sobel = SobelPlateLocate()
        plate_sobel.read_img(self._img)
        plate_sobel.set_size(20, 72)
        plate_sobel.set_gaussian_size(5)
        plate_sobel.set_morph_hw(17, 3)
        plate_sobel.set_verify_value(1, 100, 4, .5)
        plate_sobel.plate_locate()
        sobel_plates = plate_sobel.return_plates()
        sobel_regions = plate_sobel.return_regions()

        imgs, labels = self.plate_validate.main(sobel_plates)

        for i in range(len(imgs)):
            if labels[i] == IS_PLATE:
                img_plate.append(imgs[i])
                region_plate.append(sobel_regions[i])

        return img_plate, region_plate

    def __detect_plate_color(self):
        img_plate = []
        region_plate = []

        plate_color = ColorPlateLocate()
        img = plate_color.read_img(self._img)
        plate_color.set_size(20, 72)
        plate_color.set_img_hsv(255, 64, 95, 100, 140, 15, 40, 0, 30)
        plate_color.set_verify_value(1, 200, 4, .5)
        plate_color.set_morph_hw(10, 3)
        plate_color.plate_locate(img, "BLUE")
        color_plates = plate_color.return_plates()
        color_regions = plate_color.return_regions()

        imgs, labels = self.plate_validate.main(color_plates)

        for i in range(len(imgs)):
            if labels[i] == IS_PLATE:
                img_plate.append(imgs[i])
                region_plate.append(color_regions[i])

        return img_plate, region_plate

    @staticmethod
    def __draw_contours(img, plate_regions):
        for region in plate_regions:
            img = cv2.drawContours(img, [region], 0, (0, 255, 0), 2)
        return img

    @staticmethod
    def print_plate(plate):
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(plate)
        plt.show()

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, img):
        self._img = img

    @property
    def img_con_sobel(self):
        return self._img_con_sobel

    @property
    def plates_sobel(self):
        return self._plates_sobel

    @property
    def regions_sobel(self):
        return self._regions_sobel

    @property
    def img_con_color(self):
        return self._img_con_color

    @property
    def plates_color(self):
        return self._plates_color

    @property
    def regions_color(self):
        return self._regions_color

    @property
    def plate_with_no(self):
        return self._plate_with_no


if __name__ == '__main__':

    print('Image path: %s' % str(os.path.abspath('../Material')).replace('\\', '\\\\'))
    path = input('Please input your image path:')

    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    plate_rec = PlateRec()
    plate_rec.img = img

    plate_rec.main()

    for plate in plate_rec.plates_sobel:
        plate_rec.print_plate(plate)

    plate_rec.print_plate(plate_rec.img_con_sobel)

    print(plate_rec.plate_with_no)

    # for plate in plate_rec.plates_color:
    #     plate_rec.print_plate(plate)
    #
    # plate_rec.print_plate(plate_rec.img_con_color)


    # char_detect = charsSegment()
    # for img in img_plate:
    #     char_img = char_detect.read_img(img)
    #     chars = char_detect.charsSegment(char_img, 'BLUE')
    #
    #     for char in chars:
    #         plt.axis('off')
    #         plt.imshow(char)
    #         plt.show()
