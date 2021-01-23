import cv2
import parinya

cap = cv2.VideoCapture(0) #cam1-notebook
# cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) #cam2-USB
# cap = cv2.VideoCapture('Ctl1.mp4') #clip

yolo = parinya.YOLOv3('coco.names','yolov3-tiny.cfg','yolov3-tiny.weights')
while True:
    _, frame = cap.read()
    obj = yolo.detect(frame, draw=False) # yolo.detect(frame)
    
    for d in obj:
        label, left, top, width, height = d
        print(d)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)