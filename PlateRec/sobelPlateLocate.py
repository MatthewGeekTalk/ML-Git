import cv2
import numpy as np
import os


class SobelPlateLocate:
    def __init__(self):
        self.m_GaussianBlurSize = 0.
        self.img = object
        self.imgOrg = object
        self.im2 = object
        self.morphH = 0
        self.morphW = 0
        self.region = []
        self.safe_region = []
        self.safe_rect_bound = []
        self.plates = []
        self.verify_min = 0
        self.verify_max = 0
        self.verify_aspect = 0
        self.verify_error = 0
        self.rect = []
        self.safe_rect = []

    def read_img(self, img_path):
        self.img = cv2.imread(img_path)
        self.imgOrg = cv2.imread(img_path)

    def plate_locate(self):
        self.img = self.__gaussian_blur()
        self.img = self.__img_gray()
        self.img = self.__img_sobel()
        self.img = self.__img_binary()
        self.img = self.__img_morph_close()
        self.region, self.safe_region, self.rect = self.__find_plate_number_region()
        self.plates = self.__detect_region()

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

    def __sobelOper(self, m_GaussianBlurSize, morph_w, morph_h):

        return 0
    def __gaussian_blur(self):
        return cv2.GaussianBlur(self.img, (self.m_GaussianBlurSize, self.m_GaussianBlurSize), 0, 0, cv2.BORDER_DEFAULT)

    def __img_gray(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def __img_sobel(self):
        return cv2.Sobel(self.img, cv2.CV_8U, 1, 0, ksize=3)

    def __img_binary(self):
        ret, binary = cv2.threshold(self.img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        return binary

    def __img_morph_close(self):
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphW, self.morphH))
        return cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, element)

    def __find_plate_number_region(self):
        region = []
        safe_region = []
        trect = []
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
            if (rect[2] <= -5 or rect[2] >= 5):# Check angle
                safe_region.append(safe_box)
            else:
                region.append(box)
            trect.append(rect)
        return region, safe_region, trect
    # Deskew plate
    # def __deskew(self):
    #     return self
    @staticmethod
    # Enlarge Area/Region
    def __enlargeRegion(box):
        top = np.int0(box.shape[0] * 0.3)
        bottom = np.int0(box.shape[0] * 0.3)
        left = np.int0(box.shape[1] * 0.3)
        right = np.int0(box.shape[1] * 0.3)
        rect_bound = cv2.copyMakeBorder(box, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return rect_bound
    # def __enlargeRegion(self):
    #     src_height, src_width, src_channels = self.imgOrg.shape
    #     safe_region_part = []
    #     for box in self.safe_region:
    #         x = box[0, 0]
    #         y = box[0, 1]
    #         height = abs(box[2, 0] - box[0, 0])
    #         width = abs(box[0, 1] - box[1, 1])
    #         ratio = width / height
    #         if ratio > 1 and ratio < 3 and height < 120:
    #             box_part = []
    #             x_part = int(x - height * (4 - ratio))
    #             if x_part < 0:
    #                 x_part = 0
    #             width_part = int(width + height * 2 * (4 - ratio))
    #             if width_part + x_part >= src_width:
    #                 width_part = int(src_width - x_part)
    #             y_part = int(y - height * 0.08)
    #             height_part = int(height * 1.16)
    #             x0 = x_part
    #             y0 = y_part
    #             x1 = x_part
    #             y1 = y0 - width_part
    #             x2 = x0 + height_part
    #             y2 = y1
    #             x3 = x2
    #             y3 = y0
    #             box_part.append([x0, y0])
    #             box_part.append([x1, y1])
    #             box_part.append([x2, y2])
    #             box_part.append([x3, y3])
    #             safe_region_part.append(box_part)
    #     return safe_region_part

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
        for box in self.safe_region:
            cv2.drawContours(self.imgOrg, [box], 0, (0, 0, 255), 2)
            ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs)

            x1 = box[xs_sorted_index[0], 0]
            x2 = box[xs_sorted_index[3], 0]

            y1 = box[ys_sorted_index[0], 1]
            y2 = box[ys_sorted_index[3], 1]
            img_org2 = self.imgOrg.copy()
            img_plate = img_org2[y1:y2, x1:x2]
            print(img_plate.shape)
            img_large = self.__enlargeRegion(img_plate)
            self.safe_rect_bound.append(img_large)
            plates.append(img_large)

        for box in self.region:
            cv2.drawContours(self.imgOrg, [box], 0, (0, 255, 0), 2)
            ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs)

            x1 = box[xs_sorted_index[0], 0]
            x2 = box[xs_sorted_index[3], 0]

            y1 = box[ys_sorted_index[0], 1]
            y2 = box[ys_sorted_index[3], 1]
            img_org2 = self.imgOrg.copy()
            img_plate = img_org2[y1:y2, x1:x2]
            plates.append(img_plate)
        return plates

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
    # plate_locate.read_img('plate2.jpg')
    plate_locate.set_gaussian_size(5)
    plate_locate.set_morph_hw(17, 3)
    plate_locate.set_verify_value(1, 100, 4, .5)
    plate_locate.plate_locate()
    plate_locate.img_show()
