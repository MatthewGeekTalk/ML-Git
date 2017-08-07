import cv2

video_capture = cv2.VideoCapture("768x576.avi")
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if ret == False: 
        break
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    k = cv2.waitKey(100) & 0xFF
    if k== 27:
        break
video_capture.release()
cv2.destroyAllWindows()