import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('./'))
from charsSegment import charsSegment

BLUE = 'BLUE'
materials = os.listdir(os.path.abspath('../Plates/4'))

if __name__ == '__main__':
    charsSegment = charsSegment()
    for i in range(len(materials)):
        path = os.path.abspath('../Plates/4') + os.path.sep + str(materials[i])
        img = charsSegment.read_img(path)
        chars = charsSegment.charsSegment(img,BLUE)
        set_path = os.path.abspath('../trainingchar1') + os.path.sep
        for j in range(len(chars)):
            cv2.imwrite(set_path + 'char' + str(i) + '_' + str(j) + '.jpg', chars[j])