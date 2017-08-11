import cv2
import numpy as np
import os


class ColorPlateLocate:
    def __init__(self, plate_case):
        self.max_sv = 0
        self.minref_sv = 0
        self.minabs_sv = 0
        self.case = ''

        self.min_blue = 0
        self.max_blue = 0
        self.min_yellow = 0
        self.max_yellow = 0
        self.min_white = 0
        self.max_white = 0

        self.morphW = 0
        self.morphH = 0

        self.img = object
        self.imgOrg = object

        self.verify_min = 0
        self.verify_max = 0
        self.verify_aspect = 0
        self.verify_error = 0

        self.region = []
        self.safe_region = []
        self.plates = []

    def read_img(self, img_path):
        self.img = cv2.imread(img_path)
        self.imgOrg = self.img.copy()

    def set_img_hsv(self, max_sv, minref_sv, minabs_sv, min_blue, max_blue,
                    min_yellow, max_yellow, min_white, max_white):
        # min value of s and v is adaptive to h
        self.max_sv = max_sv
        self.minref_sv = minref_sv
        self.minabs_sv = minabs_sv
        # H range of blue
        self.min_blue = min_blue
        self.max_blue = max_blue
        # H range of yellow
        self.min_yellow = min_yellow
        self.max_yellow = max_yellow
        # H range of white
        self.min_white = min_white
        self.max_white = max_white

    def plate_locate(self):
        self.img = self.__set_bgr2hsv()
        if self.case == "BLUE":
            min_h = np.array([self.min_blue, 50, 50])
            max_h = np.array([self.max_blue, 255, 255])
        else:
            # no other case for now only for blue plate
            min_h = np.array([self.min_blue, 50, 50])
            max_h = np.array([self.max_blue, 255, 255])
        self.img = cv2.inRange(self.img, min_h, max_h)
        self.img = self.__img_morph_close()
        cv2.imshow('img2', self.img)
        self.region, self.safe_region = self.__find_plate_number_region()
        self.plates = self.__detect_region()

    def set_morph_hw(self, morph_w, morph_h):
        self.morphW = morph_w
        self.morphH = morph_h

    def set_verify_value(self, verify_min, verify_max, verify_aspect, verify_error):
        self.verify_min = verify_min
        self.verify_max = verify_max
        self.verify_aspect = verify_aspect
        self.verify_error = verify_error

    def img_show(self):
        cv2.imshow('img', self.imgOrg)
        try:
            for i in range(len(self.plates)):
                cv2.imshow('plates_' + str(i), self.plates[i])
        except Exception:
            pass
        cv2.waitKey(0)

    def __set_bgr2hsv(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    def __img_morph_close(self):
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphW, self.morphH))
        return cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, element)

    def __find_plate_number_region(self):
        region = []
        safe_region = []
        img_find = self.img.copy()
        im2, contours, hierarchy = cv2.findContours(img_find, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.imshow('contours', im2)
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
            region.append(box)
            safe_region.append(safe_box)
        return region, safe_region

    @staticmethod
    def __calc_safe_rect(box):
        box_reshape = np.reshape(box, (box.shape[0], box.shape[1], 1))
        x, y, w, h = cv2.boundingRect(box_reshape)
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
            plates.append(img_plate)

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


if __name__ == "__main__":
    print("Image path: %s" % str(os.path.abspath('../Material')).replace('\\', '\\\\'))
    path = input("Please input your image path:")
    plate_locate = ColorPlateLocate('BLUE')
    plate_locate.read_img(path)
    plate_locate.set_img_hsv(255, 64, 95, 100, 140, 15, 40, 0, 30)
    plate_locate.set_verify_value(1, 200, 4, .5)
    plate_locate.set_morph_hw(17, 3)
    plate_locate.plate_locate()
    plate_locate.img_show()
