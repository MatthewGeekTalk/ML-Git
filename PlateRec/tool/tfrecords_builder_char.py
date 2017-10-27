import numpy as np
import tensorflow as tf
import cv2
import os

class tfrecords_builder_char:
    def __init__(self):
        self.char = os.path.abspath('../../trainingchar1')
        self.is_0 = os.path.abspath('../../trainingchar1/0')
        self.is_1 = os.path.abspath('../../trainingchar1/1')
        self.is_2 = os.path.abspath('../../trainingchar1/2')
        self.is_3 = os.path.abspath('../../trainingchar1/3')
        self.is_4 = os.path.abspath('../../trainingchar1/4')
        self.is_5 = os.path.abspath('../../trainingchar1/5')
        self.is_6 = os.path.abspath('../../trainingchar1/6')
        self.is_7 = os.path.abspath('../../trainingchar1/7')
        self.is_8 = os.path.abspath('../../trainingchar1/8')
        self.is_9 = os.path.abspath('../../trainingchar1/9')
        self.is_A = os.path.abspath('../../trainingchar1/A')
        self.is_B = os.path.abspath('../../trainingchar1/B')
        self.is_C = os.path.abspath('../../trainingchar1/C')
        self.is_D = os.path.abspath('../../trainingchar1/D')
        self.is_E = os.path.abspath('../../trainingchar1/E')
        self.is_F = os.path.abspath('../../trainingchar1/F')
        self.is_G = os.path.abspath('../../trainingchar1/G')
        self.is_H = os.path.abspath('../../trainingchar1/H')
        self.is_J = os.path.abspath('../../trainingchar1/J')
        self.is_K = os.path.abspath('../../trainingchar1/K')
        self.is_L = os.path.abspath('../../trainingchar1/L')
        self.is_M = os.path.abspath('../../trainingchar1/M')
        self.is_N = os.path.abspath('../../trainingchar1/N')
        self.is_P = os.path.abspath('../../trainingchar1/P')
        self.is_Q = os.path.abspath('../../trainingchar1/Q')
        self.is_R = os.path.abspath('../../trainingchar1/R')
        self.is_S = os.path.abspath('../../trainingchar1/S')
        self.is_T = os.path.abspath('../../trainingchar1/T')
        self.is_U = os.path.abspath('../../trainingchar1/U')
        self.is_V = os.path.abspath('../../trainingchar1/V')
        self.is_W = os.path.abspath('../../trainingchar1/W')
        self.is_X = os.path.abspath('../../trainingchar1/X')
        self.is_Y = os.path.abspath('../../trainingchar1/Y')
        self.is_Z = os.path.abspath('../../trainingchar1/Z')
        self.is_gan = os.path.abspath('../../trainingchar1/gan')
        self.is_hu = os.path.abspath('../../trainingchar1/hu')
        self.is_jin = os.path.abspath('../../trainingchar1/jin')
        self.is_jing = os.path.abspath('../../trainingchar1/jing')
        self.is_su = os.path.abspath('../../trainingchar1/su')
        self.is_wan = os.path.abspath('../../trainingchar1/wan')
        self.is_xiang = os.path.abspath('../../trainingchar1/xiang')
        self.is_yue = os.path.abspath('../../trainingchar1/yue')
        self.is_zhe = os.path.abspath('../../trainingchar1/zhe')
        self.is_liao = os.path.abspath('../../trainingchar1/liao')
        self.is_min = os.path.abspath('../../trainingchar1/min')
        self.IS_A = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.IS_B = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_C = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_D = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_E = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_F = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_G = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_H = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_J = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_K = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_L = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_M = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_N = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_P = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_Q = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_R = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_S = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_T = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_U = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_V = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_W = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_X = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_Z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_7 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_GAN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_HU = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_JIN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.IS_JING = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.IS_SU = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.IS_WAN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        self.IS_XIANG = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        self.IS_YUE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        self.IS_ZHE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        self.IS_LIAO = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        self.IS_MIN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.TFRECORDS_ADDR = os.path.abspath('../TFRecords')

    def _list_imgs_labels(self):
        imgs = []
        labels = []
        # A-----------------------------------------------------------------------
        a = os.listdir(self.is_A)
        for i in range(len(a)):
            path = self.is_A + os.path.sep + str(a[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_A)
        # B-----------------------------------------------------------------------
        b = os.listdir(self.is_B)
        for i in range(len(b)):
            path = self.is_B + os.path.sep + str(b[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_B)
        # C-----------------------------------------------------------------------
        c = os.listdir(self.is_C)
        for i in range(len(c)):
            path = self.is_C + os.path.sep + str(c[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_C)
        # D-----------------------------------------------------------------------
        d = os.listdir(self.is_D)
        for i in range(len(d)):
            path = self.is_D + os.path.sep + str(d[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_D)
        # E-----------------------------------------------------------------------
        e = os.listdir(self.is_E)
        for i in range(len(e)):
            path = self.is_E + os.path.sep + str(e[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_E)
        # F-----------------------------------------------------------------------
        f = os.listdir(self.is_F)
        for i in range(len(f)):
            path = self.is_F + os.path.sep + str(f[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_F)
        # G-----------------------------------------------------------------------
        g = os.listdir(self.is_G)
        for i in range(len(g)):
            path = self.is_G + os.path.sep + str(g[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_G)
        # H-----------------------------------------------------------------------
        h = os.listdir(self.is_H)
        for i in range(len(h)):
            path = self.is_H + os.path.sep + str(h[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_H)
        # J-----------------------------------------------------------------------
        j = os.listdir(self.is_J)
        for i in range(len(j)):
            path = self.is_J + os.path.sep + str(j[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_J)
        # K-----------------------------------------------------------------------
        k = os.listdir(self.is_K)
        for i in range(len(k)):
            path = self.is_K + os.path.sep + str(k[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_K)
        # L-----------------------------------------------------------------------
        l = os.listdir(self.is_L)
        for i in range(len(l)):
            path = self.is_L + os.path.sep + str(l[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_L)
        # M-----------------------------------------------------------------------
        m = os.listdir(self.is_M)
        for i in range(len(m)):
            path = self.is_M + os.path.sep + str(m[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_M)
        # N-----------------------------------------------------------------------
        n = os.listdir(self.is_N)
        for i in range(len(n)):
            path = self.is_N + os.path.sep + str(n[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_N)
        # P-----------------------------------------------------------------------
        p = os.listdir(self.is_P)
        for i in range(len(p)):
            path = self.is_P + os.path.sep + str(p[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_P)
        # Q-----------------------------------------------------------------------
        q = os.listdir(self.is_Q)
        for i in range(len(q)):
            path = self.is_Q + os.path.sep + str(q[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_Q)
        # R------------------  -----------------------------------------------------
        r = os.listdir(self.is_R)
        for i in range(len(r)):
            path = self.is_R + os.path.sep + str(r[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_R)
        # S-----------------------------------------------------------------------
        s = os.listdir(self.is_S)
        for i in range(len(s)):
            path = self.is_S + os.path.sep + str(s[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_S)
        # T-----------------------------------------------------------------------
        t = os.listdir(self.is_T)
        for i in range(len(t)):
            path = self.is_T + os.path.sep + str(t[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_T)
        # U-----------------------------------------------------------------------
        u = os.listdir(self.is_U)
        for i in range(len(u)):
            path = self.is_U + os.path.sep + str(u[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_U)
        # V-----------------------------------------------------------------------
        v = os.listdir(self.is_V)
        for i in range(len(v)):
            path = self.is_V + os.path.sep + str(v[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_V)
        # W-----------------------------------------------------------------------
        w = os.listdir(self.is_W)
        for i in range(len(w)):
            path = self.is_W + os.path.sep + str(w[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_W)
        # X-----------------------------------------------------------------------
        x = os.listdir(self.is_X)
        for i in range(len(x)):
            path = self.is_X + os.path.sep + str(x[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_X)
        # Y-----------------------------------------------------------------------
        y = os.listdir(self.is_Y)
        for i in range(len(y)):
            path = self.is_Y + os.path.sep + str(y[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_Y)
        # Z-----------------------------------------------------------------------
        z= os.listdir(self.is_Z)
        for i in range(len(z)):
            path = self.is_Z + os.path.sep + str(z[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_Z)
        # 0-----------------------------------------------------------------------
        zero = os.listdir(self.is_0)
        for i in range(len(zero)):
            path = self.is_0 + os.path.sep + str(zero[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_0)
        # 1-----------------------------------------------------------------------
        one = os.listdir(self.is_1)
        for i in range(len(one)):
            path = self.is_1 + os.path.sep + str(one[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_1)
        # 2-----------------------------------------------------------------------
        two = os.listdir(self.is_2)
        for i in range(len(two)):
            path = self.is_2 + os.path.sep + str(two[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_2)
        # 3-----------------------------------------------------------------------
        three = os.listdir(self.is_3)
        for i in range(len(three)):
            path = self.is_3 + os.path.sep + str(three[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_3)
        # 4-----------------------------------------------------------------------
        four = os.listdir(self.is_4)
        for i in range(len(four)):
            path = self.is_4 + os.path.sep + str(four[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_4)
        # 5-----------------------------------------------------------------------
        five = os.listdir(self.is_5)
        for i in range(len(five)):
            path = self.is_5 + os.path.sep + str(five[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_5)
        # 6-----------------------------------------------------------------------
        six = os.listdir(self.is_6)
        for i in range(len(six)):
            path = self.is_6 + os.path.sep + str(six[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_6)
        # 7-----------------------------------------------------------------------
        seven = os.listdir(self.is_7)
        for i in range(len(seven)):
            path = self.is_7 + os.path.sep + str(seven[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_7)
        # 8-----------------------------------------------------------------------
        eight = os.listdir(self.is_8)
        for i in range(len(eight)):
            path = self.is_8 + os.path.sep + str(eight[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_8)
        # 9-----------------------------------------------------------------------
        nine = os.listdir(self.is_9)
        for i in range(len(nine)):
            path = self.is_9 + os.path.sep + str(nine[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_9)
        # 甘-----------------------------------------------------------------------
        gan = os.listdir(self.is_gan)
        for i in range(len(gan)):
            path = self.is_gan + os.path.sep + str(gan[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_GAN)
        # 沪-----------------------------------------------------------------------
        hu = os.listdir(self.is_hu)
        for i in range(len(hu)):
            path = self.is_hu + os.path.sep + str(hu[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_HU)
        # 津-----------------------------------------------------------------------
        jin = os.listdir(self.is_jin)
        for i in range(len(jin)):
            path = self.is_jin + os.path.sep + str(jin[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_JIN)
        # 京-----------------------------------------------------------------------
        jing = os.listdir(self.is_jing)
        for i in range(len(jing)):
            path = self.is_jing + os.path.sep + str(jing[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_JING)
        # 辽-----------------------------------------------------------------------
        liao = os.listdir(self.is_liao)
        for i in range(len(liao)):
            path = self.is_liao + os.path.sep + str(liao[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_LIAO)
        # 闽-----------------------------------------------------------------------
        min = os.listdir(self.is_min)
        for i in range(len(min)):
            path = self.is_min + os.path.sep + str(min[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_MIN)
        # 苏-----------------------------------------------------------------------
        su = os.listdir(self.is_su)
        for i in range(len(su)):
            path = self.is_su + os.path.sep + str(su[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_SU)
        # 皖-----------------------------------------------------------------------
        wan = os.listdir(self.is_wan)
        for i in range(len(wan)):
            path = self.is_wan + os.path.sep + str(wan[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_WAN)
        # 湘-----------------------------------------------------------------------
        xiang = os.listdir(self.is_xiang)
        for i in range(len(xiang)):
            path = self.is_xiang + os.path.sep + str(xiang[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_XIANG)
        # 粤-----------------------------------------------------------------------
        yue = os.listdir(self.is_yue)
        for i in range(len(yue)):
            path = self.is_yue + os.path.sep + str(yue[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_YUE)
        # 浙-----------------------------------------------------------------------
        zhe = os.listdir(self.is_zhe)
        for i in range(len(zhe)):
            path = self.is_zhe + os.path.sep + str(zhe[i])
            img = self._get_img(path)
            imgs.append(img)
            labels.append(self.IS_ZHE)
        return imgs, labels

    def _build_tfrecords(self, imgs, labels):
        file_name = self.TFRECORDS_ADDR + os.path.sep + 'plates_char.tfrecords'

        writer = tf.python_io.TFRecordWriter(file_name)
        labels = np.asarray(labels, dtype=np.int64)

        for i in range(len(imgs)):
            feature = {'train/label': self._int64_feature(labels[i]),
                       'train/image': self._bytes_feature(tf.compat.as_bytes(imgs[i].tostring()))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

    def main(self):
        imgs, labels = self._list_imgs_labels()
        self._build_tfrecords(imgs, labels)

    @staticmethod
    def _get_img(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        return img

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    builder = tfrecords_builder_char()
    builder.main()