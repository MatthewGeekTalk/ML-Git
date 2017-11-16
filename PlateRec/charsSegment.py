import cv2
import numpy as np
import os

class charsSegment:
    def __init__(self):
        self.rect = []
        self.img = object

    def read_img(self, img):
        # self.img = cv2.imread(img_path)
        self.img = img
        return self.img

    def charsSegment(self, img, color):
        char = []
        chars = []
        img_gay = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_threshold = self.spatial_ostu(img_gay, color)
        im2, contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.imshow('test', img_threshold)
        for i in range(len(contours)):
            pack = []
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area <= 10:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 0 and h > 0 and w < h:
                if (w + 1) > img_threshold.shape[0]:
                    rect = ((x + (w / 2), y + (h / 2)), (w + 2, h), 0)
                else:
                    rect = ((x + (w / 2), y + (h / 2)), (w + 2, h + 1), 0)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                height = abs(box[0][1] - box[2][1])
                width = abs(box[0][0] - box[2][0])
                ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
                xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
                ys_sorted_index = np.argsort(ys)
                xs_sorted_index = np.argsort(xs)

                x1 = box[xs_sorted_index[0], 0]
                x2 = box[xs_sorted_index[3], 0]

                y1 = box[ys_sorted_index[0], 1]
                y2 = box[ys_sorted_index[3], 1]
                img_plate = img_threshold[y1:y2, x1:x2]
                result = self.verifyCharSizes(img_plate, height, width)
                if result == True:
                    pack.append([x1,x2,y1,y2])
                    pack.append([height,width])
                    char.append([x1,pack])
        char = sorted(char)
        for i in range(len(char)):
            x1 = char[i][1][0][0]
            x2 = char[i][1][0][1]
            y1 = char[i][1][0][2]
            y2 = char[i][1][0][3]
            height = char[i][1][1][0]
            width = char[i][1][1][1]
            if i == 0:
                img_plate = img_threshold[0:img_threshold.shape[0], x1:x2]
            elif i == 7:
                continue
            else:
                img_plate = img_threshold[y1:y2, x1:x2]
            result = self.verifyCharSizes(img_plate, height, width)
            if img_plate.shape[0] != 0 and img_plate.shape[1] != 0:
                if i != 0:
                    if img_plate.shape[0] / img_plate.shape[1] > 2:
                        left = np.int0(img_plate.shape[1] * 0.5)
                        right = np.int0(img_plate.shape[1] * 0.5)
                        img_plate = cv2.copyMakeBorder(img_plate, 0, 0, left, right, cv2.BORDER_CONSTANT,
                                                       value=[0, 0, 0])
                img_plate = cv2.resize(img_plate, (28, 28), interpolation=cv2.INTER_CUBIC)
            if result == True:
                chars.append(img_plate)
        return chars

    def verifyCharSizes(self, img, height, width):
        aspect = 0.5
        charAspect = width / height
        error = 0.7
        minHeight = 10
        maxHeight = 50
        # We have a different aspect ratio for number 1, and it can be ~0.2
        minAspect = 0.05
        maxAspect = aspect + aspect * error
        area = cv2.countNonZero(img)
        bbArea = height * width
        percPixels = area / bbArea
        if (percPixels <= 1 and charAspect > minAspect and charAspect < maxAspect and
                    height >= minHeight and height <= maxHeight):
            return True
        else:
            return False

    def spatial_ostu(self, src, color):
        if color == 'BLUE':
            ret, img_thre = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        elif color == 'YELLOW':
            ret, img_thre = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        elif color == 'WHITE':
            ret, img_thre = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        else:
            ret, img_thre = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        return img_thre


if __name__ == '__main__':
    img = cv2.imread('char0.jpg')
    charsSegment = charsSegment()
    img = charsSegment.read_img(img)
    chars = charsSegment.charsSegment(img, 'BLUE')
    for i in range(len(chars)):
        cv2.imshow('plates_' + str(i), chars[i])
        set_path = os.path.abspath('../trainingchar1') + os.path.sep
        cv2.imwrite(set_path + 'char' + str(i) + '.jpg', chars[i])
    cv2.waitKey(0)
