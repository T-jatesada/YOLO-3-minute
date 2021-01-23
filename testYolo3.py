import cv2
import parinya
import pafy #pip install pafy # pip install --upgrade youtube_dl

url = 'https://youtu.be/iv89oPjBkGY'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4") # (preftype="webm") not work!
cap = cv2.VideoCapture(play.url)

yolo = parinya.YOLOv3('coco.names','yolov3-tiny.cfg','yolov3-tiny.weights')
while True:
    _, frame = cap.read()
    obj = yolo.detect(frame) #obj = yolo.detect(frame, draw=False)
    for d in obj:
        label, left, top, width, height =d
        print(d)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)