import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('./'))
from sobelPlateLocate import SobelPlateLocate
from colorPlateLocate import ColorPlateLocate


BLUE = 'BLUE'

# plates = []
# materials = os.listdir(os.path.abspath('../Test_img'))
#
# plate_color_digger = ColorPlateLocate()
# plate_color_digger.set_size(20, 70)
# plate_color_digger.set_img_hsv(255, 64, 95, 100, 140, 15, 40, 0, 30)
# plate_color_digger.set_verify_value(1, 200, 4, .5)
# plate_color_digger.set_morph_hw(10, 3)
# for i in range(len(materials)):
#     path = os.path.abspath('../Test_img') + os.path.sep + str(materials[i])
#     plate_color_digger.init_plates()
#     plate_color_digger.plate_locate(plate_color_digger.read_img(path), BLUE)
#     plates = plate_color_digger.return_plates()
#     set_path = os.path.abspath('../trainingSetColor') + os.path.sep
#     for j in range(len(plates)):
#         cv2.imwrite(set_path + 'plate' + str(i) + '_' + str(j) + '.jpg', plates[j])


plates = []
materials = os.listdir(os.path.abspath('../Test_img'))

plate_sobel_digger = SobelPlateLocate()
plate_sobel_digger.set_size(20, 70)
plate_sobel_digger.set_gaussian_size(5)
plate_sobel_digger.set_morph_hw(17, 3)
plate_sobel_digger.set_verify_value(1, 100, 4, .5)
for i in range(len(materials)):
    path = os.path.abspath('../Test_img') + os.path.sep + str(materials[i])
    plate_sobel_digger.read_img(path)
    plates = plate_sobel_digger.plate_locate()
    set_path = os.path.abspath('../trainingSetsobel') + os.path.sep
    print(set_path)
    for j in range(len(plates)):
        cv2.imwrite(set_path + 'platesobel' + str(i) + '_' + str(j) + '.jpg', plates[j])

