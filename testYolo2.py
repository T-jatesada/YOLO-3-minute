import cv2
import parinya

cap = cv2.VideoCapture('clip1.mp4')
yolo = parinya.YOLOv3('coco.names','yolov3-tiny.cfg','yolov3-tiny.weights')
while True:
    _, frame = cap.read()
    obj = yolo.detect(frame) #obj = yolo.detect(frame, draw=False)
    for d in obj:
        label, left, top, width, height =d
        print(d)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)