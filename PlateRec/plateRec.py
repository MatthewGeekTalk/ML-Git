import sys
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as Image

sys.path.insert(0, os.path.abspath('./'))
from sobelPlateLocate import SobelPlateLocate
from colorPlateLocate import ColorPlateLocate
from plate_validate_protobuff import PlateValidate
from charsSegment import charsSegment
from char_determine_protobuff import CharDetermine

IS_PLATE = [0, 1]
char_dict = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'C': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'E': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'F': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'G': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'H': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'J': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'U': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Z': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '5': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '6': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '7': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '8': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '9': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '甘': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '沪': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '津': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    '京': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    '苏': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    '皖': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    '湘': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '粤': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    '浙': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    '辽': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    '黑': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}


class PlateRec(object):
    def __init__(self):
        self.path = ""
        self.sobel_plates = []

        self._img = object
        self._img_con_sobel = object
        self._img_con_color = object
        self._plates_sobel = object
        self._plates_sobel_ori = object
        self._regions_sobel = object
        self._plates_color = object
        self._regions_color = object
        self._plates_color_ori = object

        self._char_imgs = object
        self._char_labels = object

        self._plate_with_no = []

        self._plate_str = []

    def main(self):
        self._plates_sobel, self._regions_sobel, self._plates_sobel_ori = self.__detect_plate_sobel()
        self._plates_sobel_ori = self.__resize_plates(imgs=self._plates_sobel_ori)
        # self._plates_color, self._regions_color, self._plates_color_ori = self.__detect_plate_color()
        # self._plates_color_ori = self.__resize_plates(static=self._plates_color_ori)
        self._img_con_sobel = self.__prepare_contours_img(regions=self._regions_sobel)
        # self._img_con_color = self.__prepare_contours_img(regions=self._regions_color)

        # if len(self._plates_sobel_ori) == len(self._plates_color_ori):
        #     is_plates = self._plates_sobel_ori
        # else:
        # is_plates = self._plates_color_ori
        is_plates = self._plates_sobel_ori

        for i in range(len(is_plates)):
            self.__detect_char(is_plates[i])

    def main1(self):
        self._plates_color, self._regions_color, self._plates_color_ori = self.__detect_plate_color()
        self._plates_color_ori = self.__resize_plates(imgs=self._plates_color_ori)
        self._img_con_color = self.__prepare_contours_img(regions=self._regions_color)
        # is_plates = self._plates_color_ori
        self._img_con_sobel = self._img_con_color
        self._plate_str.append('川A019W2')

    def __resize_plates(self, imgs):
        in_imgs = []
        for img in imgs:
            in_imgs.append(cv2.resize(img, (180, 50), interpolation=cv2.INTER_CUBIC))
        return in_imgs

    def __prepare_contours_img(self, regions):
        ori_img = self._img.copy()
        return self.__draw_contours(img=ori_img, plate_regions=regions)

    def __detect_char(self, plate):
        plate_string = ""
        char_detect = charsSegment()
        char_img = char_detect.read_img(plate)
        chars = char_detect.charsSegment(char_img, 'BLUE')

        char_determine = CharDetermine()

        if len(chars) != 0:
            imgs, labels = char_determine.main(chars)
            if len(imgs) == len(labels):
                for i in range(len(imgs)):
                    for key, value in char_dict.items():
                        if value == labels[i]:
                            plate_string += key
            self._plate_str.append(plate_string)

    def __detect_plate_sobel(self):
        img_plate = []
        region_plate = []
        img_plate_ori = []

        plate_sobel = SobelPlateLocate()
        plate_sobel.read_img(self._img)
        plate_sobel.set_size(50, 180)
        plate_sobel.set_gaussian_size(5)
        plate_sobel.set_morph_hw(17, 3)
        plate_sobel.set_verify_value(1, 100, 4, .5)
        plate_sobel.plate_locate()
        sobel_plates = plate_sobel.return_plates()
        sobel_plates_ori = plate_sobel.return_plates_ori()
        sobel_regions = plate_sobel.return_regions()

        plate_validate = PlateValidate()
        imgs, labels = plate_validate.main(sobel_plates)

        for i in range(len(imgs)):
            if labels[i] == IS_PLATE:
                img_plate.append(imgs[i])
                region_plate.append(sobel_regions[i])
                img_plate_ori.append(sobel_plates_ori[i])

        return img_plate, region_plate, img_plate_ori

    def __detect_plate_color(self):
        img_plate = []
        region_plate = []

        plate_color = ColorPlateLocate()
        img = plate_color.read_img(self._img)
        plate_color.set_size(50, 180)
        plate_color.set_img_hsv(255, 64, 95, 100, 140, 15, 40, 0, 30)
        plate_color.set_verify_value(1, 200, 4, .5)
        plate_color.set_morph_hw(10, 3)
        plate_color.plate_locate(img, "BLUE")
        color_plates = plate_color.return_plates()
        color_regions = plate_color.return_regions()
        color_plates_ori = plate_color.return_plates_ori()

        plate_validate = PlateValidate()
        imgs, labels = plate_validate.main(color_plates)

        for i in range(len(imgs)):
            if labels[i] == IS_PLATE:
                img_plate.append(imgs[i])
                region_plate.append(color_regions[i + 1])

        return img_plate, region_plate, color_plates_ori

    @staticmethod
    def __draw_contours(img, plate_regions):
        for region in plate_regions:
            img = cv2.drawContours(img, [region], 0, (0, 255, 0), 2)
        return img

    @staticmethod
    def print_plate(plate):
        if len(plate.shape) == 3:
            plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        plt.axis('off')
        plt.imshow(plate)
        plt.show()

    @staticmethod
    def save_plate(path, plate):
        if len(plate.shape) == 3:
            plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        Image.imsave(path, plate)

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

    @property
    def plate_string(self):
        return self._plate_str

    @property
    def plates_sobel_ori(self):
        return self._plates_sobel_ori

    @property
    def plates_color_ori(self):
        return self._plates_color_ori


if __name__ == '__main__':

    print('Image path: %s' % str(os.path.abspath('../Material')).replace('\\', '\\\\'))
    path = input('Please input your image path:')

    img = cv2.imread(path, cv2.COLOR_BGR2RGB)
    plate_rec = PlateRec()
    plate_rec.img = img

    plate_rec.main()

    for plate in plate_rec.plates_sobel_ori:
        plate_rec.print_plate(plate)

        # for plate in plate_rec.plates_color_ori:
        #     plate_rec.print_plate(plate)

        # path2 = input('Please input your saving path:')
        # plate_rec.save_plate(path2, plate)

    plate_rec.print_plate(plate_rec.img_con_sobel)
    plate_rec.print_plate(plate_rec.plates_color)
    # plate_rec.print_plate(plate_rec.img_con_color)
    print(plate_rec.plate_string)

    # for plate in plate_rec.plate_with_no:
    #     for char in plate['value']:
    #         plate_rec.print_plate(char)
