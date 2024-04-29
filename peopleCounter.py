from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

model = YOLO('./yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# cap = cv2.VideoCapture(1)
# mask = cv2.imread("mask.png")
# mask = cv2.resize(mask,(1280,720))
cap = cv2.VideoCapture("./football.mp4")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# cap.set(3,950)
# cap.set(4,480)

while True:
    success, img = cap.read()
    # maskimg = cv2.bitwise_and(img, mask)
    # maskimg = img & mask
    results = model(img,stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person":
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),scale=1, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))



            # print(x1, y1, x2, y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)
    resultsTracker = tracker.update(detections)
    print(resultsTracker)

    count = []
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        # print(result)
        w, h = x2 - x1, y2 - y1
        # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), 1)
        cv2.putText(img, str(id), (x1,y1),cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 1)
        # cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),scale=1, thickness=1, offset=5)
        if count.count(id) == 0:
            count.append(id)

    cv2.putText(img, str(len(count)), (255, 320), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 255), 4)


    cv2.imshow('video',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cap.destroyAllWindows()