import cv2
import numpy as np
import math
import os


class SobelPlateLocate:
    def __init__(self):
        self.m_GaussianBlurSize = 0.
        self.m_angle = 60
        self.img = object
        self.imgOrg = object
        self.im2 = object
        self.img_opr = object
        self.morphH = 0
        self.morphW = 0
        self.region = []
        self.plates = []
        self.verify_min = 0
        self.verify_max = 0
        self.verify_aspect = 0
        self.verify_error = 0
        self.rect = []
        self.angle = []
        self.height = 0
        self.width = 0

    def read_img(self, img_path):
        self.img = cv2.imread(img_path)
        self.imgOrg = cv2.imread(img_path)

    def plate_locate(self):
        self.img = self.__gaussian_blur(self.img, self.m_GaussianBlurSize)
        self.img = self.__img_gray(self.img)
        self.img = self.__img_sobel(self.img)
        self.img = self.__img_binary(self.img)
        self.img = self.__img_morph_close(self.img,self.morphW, self.morphH)
        self.region, self.angle = self.__find_plate_number_region()
        # self.img_opr = self.__sobelOper(self.img_opr, 3, 10, 3)
        self.plates = self.__detect_region()
        return self.plates

    def set_size(self, height, width):
        self.height = height
        self.width = width

    def set_gaussian_size(self, gaussian_blur_size):
        self.m_GaussianBlurSize = gaussian_blur_size

    def set_morph_hw(self, morph_w, morph_h):
        self.morphW = morph_w
        self.morphH = morph_h

    def set_verify_value(self, verify_min, verify_max, verify_aspect, verify_error):
        self.verify_min = verify_min
        self.verify_max = verify_max
        self.verify_aspect = verify_aspect
        self.verify_error = verify_error

    def __sobelOper(self, src, m_GaussianBlurSize, morph_w, morph_h):
        img_opr = src.copy()
        img_opr = cv2.GaussianBlur(img_opr, (m_GaussianBlurSize, m_GaussianBlurSize), 0, 0, cv2.BORDER_DEFAULT)
        if img_opr.shape[2] == 3:
            img_opr = cv2.cvtColor(img_opr, cv2.COLOR_BGR2GRAY)
        img_opr = self.__img_sobel(img_opr)
        img_opr = self.__img_binary(img_opr)
        img_opr = self.__img_morph_close(img_opr, morph_w, morph_h)
        return img_opr
    def __gaussian_blur(self, img, m_GaussianBlurSize):
        return cv2.GaussianBlur(img, (m_GaussianBlurSize, m_GaussianBlurSize), 0, 0, cv2.BORDER_DEFAULT)

    def __img_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def __img_sobel(self, img):
        img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
        img = cv2.convertScaleAbs(img)
        return cv2.addWeighted(img, 1, 0, 0, 0)

    def __img_binary(self, img):
        ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        return binary

    def __img_morph_close(self, img, morphW, morphH):
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (morphW, morphH))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)

    def __resize_plates(self, img):
        return cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

    def __find_plate_number_region(self):
        region = []
        angle = []
        img_find = self.img.copy()
        im2, contours, hierarchy = cv2.findContours(img_find, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area == 0:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            height = int(rect[1][1])
            width = int(rect[1][0])
            if height == 0 or width == 0:
                continue
            if self.__verify_value(height, width):
                continue
            err, safe_rect = self.__calc_safe_rect(box)
            if not err:
                continue
            safe_box = cv2.boxPoints(safe_rect)
            safe_box = np.int0(safe_box)
            safe_height = abs(safe_box[0][1] - safe_box[2][1])
            safe_width = abs(safe_box[0][0] - safe_box[2][0])
            if self.__verify_value(safe_height, safe_width):
                continue
            region.append(safe_box)
            if height > width :
                angle1 = rect[2] + 90
            else:
                angle1 = rect[2]
            center_angle = [safe_rect[0][0], safe_rect[0][1], angle1]
            angle.append(center_angle)
        return region, angle
    #Enlarge and Rotation
    def __enlarge_rotation(self, src, angle):
        # 增大图片边缘pedding
        size_original = (src.shape[1], src.shape[0])
        img_opr = self.__enlargeRegion(src)
        # 角度大于5度的，首先需要旋转
        size_enlarge = (img_opr.shape[1], img_opr.shape[0])
        center = (img_opr.shape[1] / 2, img_opr.shape[0] / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_opr = cv2.warpAffine(img_opr, M, size_enlarge, cv2.INTER_CUBIC)
        # 剪裁
        img_opr = cv2.getRectSubPix(img_opr, size_original, center)
        return img_opr
    # Deskew plate
    def __deskew(self, src, angle, src_sc):
        if (src.shape[1] != 0 and src.shape[0] != 0):
            img_opr = self.__enlarge_rotation(src, angle)
        else:
            return src_sc
        if (src.shape[1] != 0 and src.shape[0] != 0):
            img_opr_sc = self.__enlarge_rotation(src_sc, angle)
        else:
            return src_sc
        slope = 0
        err, slope = self.__isdeflection(img_opr, angle, slope)
        if (err):
            img_opr = self.__affine(img_opr_sc, slope)
        else:
            img_opr = img_opr_sc
            # print("Affine is not needed")
        return img_opr
    #仿射变换
    def __affine(self,src, slope):
        dstTri = [(0, 0), (0, 0), (0, 0)]
        plTri = [(0, 0), (0, 0), (0, 0)]
        height = src.shape[0]
        width = src.shape[1]
        xiff = abs(slope) * height
        if slope > 0:
            #right, new position is xiff/2
            plTri[0] = (0, 0)
            plTri[1] = (width - xiff - 1, 0)
            plTri[2] = (0 + xiff, height - 1)

            dstTri[0] = (xiff / 2, 0)
            dstTri[1] = (width - 1 - xiff / 2, 0)
            dstTri[2] = (xiff / 2, height - 1)
        else:
            #left, new position is -xiff/2
            plTri[0] = (0 + xiff, 0)
            plTri[1] = (width - 1, 0)
            plTri[2] = (0, height - 1)

            dstTri[0] = (xiff / 2, 0)
            dstTri[1] = (width - 1 - xiff + xiff / 2, 0)
            dstTri[2] = (xiff / 2, height - 1)
        M = cv2.getAffineTransform(np.float32(plTri), np.float32(dstTri))
        if (height > 36 or width > 136):
            dst = cv2.warpAffine(src, M, (width, height), cv2.INTER_AREA)
        else:
            dst = cv2.warpAffine(src, M, (width, height), cv2.INTER_CUBIC)
        return dst
    # Check plate if need affine
    def __isdeflection(self, src, angle, slope):
        height = src.shape[0]
        width = src.shape[1]
        comp_index = [int(height / 4), int(height / 4 * 2), int(height / 4 * 3)]
        len = [0, 0, 0]

        for i in range(0, 2):
            row = comp_index[i]
            value = 0
            j = 0
            while (0 == value and j < width):
                value = src[row][j]
                j = j + 1
            len[i] = j

        maxlen = max(len[2], len[0])
        minlen = min(len[2], len[0])
        difflen = abs(len[2] - len[0])
        PI = 3.14159265
        g = math.tan(angle * PI / 180.0)
        if (maxlen - len[1] > width / 32 or len[1] - minlen > width / 32):
            slope_can_1 = (len[2] - len[0]) / comp_index[1]
            slope_can_2 = (len[1] - len[0]) / (comp_index[0])
            slope_can_3 = (len[2] - len[1]) / (comp_index[0])
            slope = slope_can_1 if abs(slope_can_1 - g) <= abs(slope_can_2 - g) else slope_can_2
            return True, slope
        else:
            slope = 0
        return False, slope
    @staticmethod
    # Enlarge Area/Region
    def __enlargeRegion(box):
        top = np.int0(box.shape[0] * 0.1)
        bottom = np.int0(box.shape[0] * 0.1)
        left = np.int0(box.shape[1] * 0.1)
        right = np.int0(box.shape[1] * 0.1)
        rect_bound = cv2.copyMakeBorder(box, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return rect_bound
    @staticmethod
    def __calc_safe_rect(box):
        box_reshape = np.reshape(box, (box.shape[0], box.shape[1], 1))
        x, y, w, h = cv2.boundingRect(box_reshape)
        if w < 0 or h < 0 or w < h:
            return False, ((x + (w / 2), y + (h / 2)), (w, h), 0)
        else:
            return True, ((x + (w / 2), y + (h / 2)), (w, h), 0)

    def __verify_value(self, height, width):
        error = self.verify_error
        aspect = self.verify_aspect
        vmin = 34 * 8 * self.verify_min
        vmax = 34 * 8 * self.verify_max
        rmin = aspect - aspect * error
        rmax = aspect + aspect * error
        area = float(height) * float(width)
        r = float(width) / float(height)
        if r < 1:
            r = float(height) / float(width)
        if (area < vmin or area > vmax) or (r < rmin or r > rmax):
            return True
        else:
            return False

    def __detect_region(self):
        plates = []
        i = 0
        for box in self.region:
            # cv2.drawContours(self.imgOrg, [box], 0, (0, 255, 0), 2)
            ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs)

            x1 = box[xs_sorted_index[0], 0]
            x2 = box[xs_sorted_index[3], 0]

            y1 = box[ys_sorted_index[0], 1]
            y2 = box[ys_sorted_index[3], 1]
            img_org = self.img.copy()
            img_org_orginal = self.imgOrg.copy()
            img_plate = img_org[y1:y2, x1:x2]
            img_plate_sc = img_org_orginal[y1:y2, x1:x2]
            if (img_plate.shape[1] != 0 and img_plate.shape[0] != 0 and
                img_plate_sc.shape[1] != 0 and img_plate_sc.shape[0] != 0):
                if (self.angle[i][2] < -5 or self.angle[i][2] > 5):
                    img_plate_sc = self.__deskew(img_plate, self.angle[i][2], img_plate_sc)
                img_plate_sc = self.__resize_plates(img_plate_sc)
                plates.append(img_plate_sc)
            i = i + 1
        return plates

    def return_plates(self):
        return self.plates

    def img_show(self):
        cv2.imshow('img', self.imgOrg)
        try:
            for i in range(len(self.plates)):
                cv2.imshow('plates_' + str(i), self.plates[i])
        except Exception:
            pass
        cv2.waitKey(0)


if __name__ == '__main__':
    print('Image path: %s' % str(os.path.abspath('../Material')).replace('\\', '\\\\'))
    path = input('Please input your image path:')
    plate_locate = SobelPlateLocate()
    plate_locate.read_img(path)
    plate_locate.set_size(20, 72)
    plate_locate.set_gaussian_size(5)
    plate_locate.set_morph_hw(17, 3)
    plate_locate.set_verify_value(1, 100, 4, .5)
    plates = plate_locate.plate_locate()
    plate_locate.img_show()
