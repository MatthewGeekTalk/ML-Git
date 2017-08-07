import cv2
import numpy as np
import copy

video_capture = cv2.VideoCapture("vedio.avi")
foreGround = np.zeros((288, 384, 3), np.uint8)
pBgModel = cv2.createBackgroundSubtractorMOG2()
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if ret == False:
        break
    image = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    fgmask = pBgModel.apply(image)
    cv2.GaussianBlur(fgmask,(5,5),0)
    cv2.threshold(fgmask,10,255,cv2.THRESH_BINARY)
    foreGround = copy.copy(image)
    foreGround = cv2.bitwise_and(foreGround, foreGround, mask = fgmask)
    backGround = pBgModel.getBackgroundImage()
    cv2.imshow('Video', image)
    cv2.imshow("Background", backGround)
    cv2.imshow("ForeGround", foreGround)
    cv2.imshow("Foreground Mask", fgmask)
    # Hit 'q' on the keyboard to quit!
    k = cv2.waitKey(100) & 0xFF
    if k== 27:
        break
video_capture.release()
cv2.destroyAllWindows()