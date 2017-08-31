import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('./'))
from sobelPlateLocate import SobelPlateLocate
from colorPlateLocate import ColorPlateLocate

BLUE = 'BLUE'

plates = []
materials = os.listdir(os.path.abspath('../Training'))

plate_color_digger = ColorPlateLocate()
plate_color_digger.set_size(50, 180)
plate_color_digger.set_img_hsv(255, 64, 95, 100, 140, 15, 40, 0, 30)
plate_color_digger.set_verify_value(1, 200, 4, .5)
plate_color_digger.set_morph_hw(10, 3)
for i in range(len(materials)):
    path = os.path.abspath('../Training') + os.path.sep + str(materials[i])
    plate_color_digger.init_plates()
    plate_color_digger.plate_locate(plate_color_digger.read_img(path), BLUE)
    plates = plate_color_digger.return_plates()
    set_path = os.path.abspath('../trainingSetColor') + os.path.sep
    for j in range(len(plates)):
        cv2.imwrite(set_path + 'plate' + str(i) + '_' + str(j) + '.jpg', plates[j])

