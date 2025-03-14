import cv2
import cvzone
from ultralytics import YOLO

from sort import *

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../videos/cars.mp4')
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('../Yolo-Weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
# massking image import
mask = cv2.imread("r.png")
# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [180, 400, 673,400]
totalCount = []

while True:
    success, img = cap.read()
    results = model(img, stream=True, )
    imgRegion = cv2.bitwise_and(img, mask)  # overlay the mask in the video
    results = model(imgRegion, stream=True)  # only take that rusuts in thath area
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)#drowing a line in the road to takake count from it
    detections = np.empty(
        (0, 5))  # trakin input mus be a numpy arry with x1,y1,x2,y2,confidence level for that we build a empty array

    for r in results:  # builing a box around recnazed objects
        boxes = r.boxes
        for box in boxes:
            # cvbox
            # x1,y1,x2,y2 = box.xyxy[0]#box codinats
            # x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # cvzone box
            x1, y1, x2, y2 = box.xyxy[0]  # box codinats
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            bbox = int(x1), int(y1), int(w), int(h)

            conf = float(box.conf[0])  # confidet level of class
            conf = round(conf, 2)  # rounding the value

            # class how many wehicals ver count i waas need tb shown in video

            cls = int(box.cls[0])  # this wil;l give tha class id using that w have to find the class
            currrentclass = classNames[cls]

            if currrentclass in ['truck','car','bus'] and conf > 0.5:  # we are taking only bus, car, truck and conf > 0.5:
                cvzone.putTextRect(img, f"{currrentclass} {conf}", (max(0, x1), max(30, y1)), scale=0.8, thickness=1,offset=2)
                # capuruing data as a arry to entr to the tracker
                currentArray3 = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray3))  # stack them on 0 arry we creat


    resultsTracker = tracker.update(detections)  # enter the data thath need to be track the car then as the output in that tracker it givs x1,y1,x2,y2,id of thath class
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w,h=x2-x1,y2-y1
        print(result)
        bbox = x1, y1, w, h
        cvzone.cornerRect(img, bbox,t=2)  # cvzone box worke with box.xyxy[0],but bbox = x1,y1 ,w,h where w = x2-x1 , h=y2-y1
        cvzone.putTextRect(img, f"{id}", (max(0, x1), max(30, y2)), scale=1, thickness=1,offset=1)  # disply data in video fream



        cx,cy = x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-20 < cy < limits[1]+20 :
            if totalCount.count(id)==0:#we will make a list of all athe  count and if an id has count multipul times it will not be add to the couter list
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f"{len(totalCount)}", (50, 50), scale=0.8, thickness=1,
                       offset=2)



    cv2.imshow('image', img)
    cv2.waitKey(1)
