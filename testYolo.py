import cv2
import parinya #pip install requests pillow parinya

cap = cv2.VideoCapture(0) #cam1-notebook
# cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) #cam2-USB
# cap = cv2.VideoCapture('Ctl1.mp4') #clip

yolo = parinya.YOLOv3('coco.names','yolov3-tiny.cfg','yolov3-tiny.weights')
while True:
    _, frame = cap.read()
    yolo.detect(frame)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)