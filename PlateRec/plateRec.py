import sys
import os
import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('./'))
from sobelPlateLocate import SobelPlateLocate
from plate_validate import PlateValidate


# BLUE = 'BLUE'
# YELLOW = 'YELLOW'
# WHITE = 'WHITE'

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

    print('Image path: %s' % str(os.path.abspath('../Material')).replace('\\', '\\\\'))
    path = input('Please input your image path:')

    # get plates from sobel
    plate_sobel = SobelPlateLocate()
    plate_sobel.read_img(path)
    plate_sobel.set_size(20, 72)
    plate_sobel.set_gaussian_size(5)
    plate_sobel.set_morph_hw(17, 3)
    plate_sobel.set_verify_value(1, 100, 4, .5)
    plate_sobel.plate_locate()
    sobel_plates = plate_sobel.return_plates()

    plate_validate = PlateValidate(sobel_plates)
    imgs, labels = plate_validate.main()

    for i in range(len(imgs)):
        plt.axis('off')
        plt.imshow(imgs[i])
        plt.show()
        print(labels[i])





        # # get plates from color
        # plate_color = ColorPlateLocate()
        # plate_color.set_img_hsv(255, 64, 95, 100, 140, 15, 40, 0, 30)
        # plate_color.set_verify_value(1, 200, 4, .5)
        # plate_color.set_morph_hw(10, 3)
        # plate_color.plate_locate(plate_color.read_img(path), BLUE)
        # color_plates = plate_color.return_plates()
