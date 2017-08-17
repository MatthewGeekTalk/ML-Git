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
        self.rect = []
        self.safe_rect = []
        self.plates = []

    def read_img(self, img_path):
        self.img = cv2.imread(img_path)
        self.imgOrg = self.img.copy()
        return self.img

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

    def __set_min_max(self, case):
        if case == "BLUE":
            min_h = np.array([self.min_blue, 50, 50])
            max_h = np.array([self.max_blue, 255, 255])
        elif case == "YELLOW":
            min_h = np.array([26, 43, 46])
            max_h = np.array([34, 255, 255])
        elif case == "WHITE":
            min_h = np.array([0, 0, 221])
            max_h = np.array([180, 30, 255])
        else:
            #     no input, using blue
            min_h = np.array([self.min_blue, 50, 50])
            max_h = np.array([self.max_blue, 255, 255])
        return min_h, max_h

    def plate_locate(self, img, case):
        img = self.__set_bgr2hsv(img)
        min_h, max_h = self.__set_min_max(case)
        img = cv2.inRange(img, min_h, max_h)
        img = self.__img_morph_close(img)
        cv2.imshow('img2', img)
        region = self.__find_plate_number_region(img)
        plates = self.__detect_region(region, self.imgOrg)
        plates = self.__split_plate(plates, case)
        self.plates = plates

    def set_morph_hw(self, morph_w, morph_h):
        self.morphW = morph_w
        self.morphH = morph_h

    def __split_plate(self, plates, case):
        split_plates = []
        for plate in plates:
            img_org = plate.copy()
            plate = self.__set_bgr2hsv(plate)
            min_h, max_h = self.__set_min_max(case)
            plate = cv2.inRange(plate, min_h, max_h)
            plate = self.__img_morph_close(plate)
            img_binary = plate.copy()
            im2, contours, hierarchy = cv2.findContours(plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for i in range(len(contours)):
                cnt = contours[i]
                rect = cv2.minAreaRect(cnt)
                height = int(rect[1][1])
                width = int(rect[1][0])
                if height == 0 or width == 0:
                    continue
                if self.__verify_value(height, width):
                    continue
                if self.__check_angle(height, width, rect[2]):
                    continue
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                err, safe_rect = self.__calc_safe_rect(box, plate)
                if not err:
                    continue
                safe_box = cv2.boxPoints(safe_rect)
                safe_box = np.int0(safe_box)
                ys = [safe_box[0, 1], safe_box[1, 1], safe_box[2, 1], safe_box[3, 1]]
                xs = [safe_box[0, 0], safe_box[1, 0], safe_box[2, 0], safe_box[3, 0]]
                ys_sorted_index = np.argsort(ys)
                xs_sorted_index = np.argsort(xs)

                x1 = safe_box[xs_sorted_index[0], 0]
                x2 = safe_box[xs_sorted_index[3], 0]

                y1 = safe_box[ys_sorted_index[0], 1]
                y2 = safe_box[ys_sorted_index[3], 1]
                # bi_plate = img_binary[y1:y2, x1:x2]
                # bi_plate = bi_plate / 255
                img_plate = img_org[y1:y2, x1:x2]
                split_plates.append(img_plate)
        return split_plates

    @staticmethod
    def __calc_parallelogram(self, img):
        print(img)

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
        # cv2.destroyAllWindows()

    @staticmethod
    def __set_bgr2hsv(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def __img_morph_close(self, img):
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphW, self.morphH))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)

    def __find_plate_number_region(self, img):
        region = []
        img_find = img.copy()
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
            if self.__check_angle(height, width, rect[2]):
                continue
            err, safe_rect = self.__calc_safe_rect(box, img)
            if not err:
                continue
            angel = (safe_rect, rect)
            region.append(angel)
        return region

    @staticmethod
    def __check_angle(height, width, angle):
        if width / height < 1:
            angle = angle + 90
        if angle > 60 or angle < -60:
            return True
        else:
            return False

    @staticmethod
    def __calc_safe_rect(box, img):
        box_reshape = np.reshape(box, (box.shape[0], box.shape[1], 1))
        x, y, w, h = cv2.boundingRect(box_reshape)
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if w > img.shape[1]:
            w = img.shape[1] - 1
        if h > img.shape[0]:
            h = img.shape[0] - 1
        if w * h == 0:
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

    @staticmethod
    def __rotate_img(rect, img):
        height = int(rect[1][1])
        width = int(rect[1][0])
        if height > width:
            angel = rect[2] + 90
        else:
            angel = rect[2]
        rotation_m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[1] / 2), angel, 1)
        img = cv2.warpAffine(img, rotation_m, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
        return img

    def __detect_region(self, region, img):
        plates = []
        for angel in region:
            safe_rect = angel[0]
            rect = angel[1]
            safe_box = cv2.boxPoints(safe_rect)
            safe_box = np.int0(safe_box)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(self.imgOrg, [safe_box], 0, (0, 0, 255), 2)
            cv2.drawContours(self.imgOrg, [box], 0, (0, 255, 0), 2)
            ys = [safe_box[0, 1], safe_box[1, 1], safe_box[2, 1], safe_box[3, 1]]
            xs = [safe_box[0, 0], safe_box[1, 0], safe_box[2, 0], safe_box[3, 0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs)

            x1 = safe_box[xs_sorted_index[0], 0]
            x2 = safe_box[xs_sorted_index[3], 0]

            y1 = safe_box[ys_sorted_index[0], 1]
            y2 = safe_box[ys_sorted_index[3], 1]
            img_plate = img[y1:y2, x1:x2]

            img_plate = self.__enlarge_region(img_plate)
            img_plate = self.__rotate_img(rect, img_plate)
            plates.append(img_plate)
        return plates

    @staticmethod
    def __enlarge_region(plate):
        top = np.int0(plate.shape[0] * 0.3)
        bottom = np.int0(plate.shape[0] * 0.3)
        left = np.int0(plate.shape[1] * 0.3)
        right = np.int0(plate.shape[1] * 0.3)
        rect_bound = cv2.copyMakeBorder(plate, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return rect_bound


if __name__ == "__main__":
    print("Image path: %s" % str(os.path.abspath('../Material') + os.path.sep).replace('\\', '\\\\'))
    path = input("Please input your image path:")
    plate_locate = ColorPlateLocate('BLUE')
    my_img = plate_locate.read_img(path)
    plate_locate.set_img_hsv(255, 64, 95, 100, 140, 15, 40, 0, 30)
    plate_locate.set_verify_value(1, 200, 4, .5)
    plate_locate.set_morph_hw(10, 3)
    plate_locate.plate_locate(my_img, "BLUE")
    plate_locate.img_show()
