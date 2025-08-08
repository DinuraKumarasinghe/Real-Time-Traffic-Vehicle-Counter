import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from sort import Sort

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Video capture
cap = cv2.VideoCapture('../videos/cars.mp4')
cap.set(3, 1280)
cap.set(4, 720)

# YOLO model
model = YOLO('../Yolo-Weights/yolov8m.pt')  # Use medium model to reduce memory

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

# Mask image (optional)
mask = cv2.imread("r.png")
if mask is not None:
    mask = cv2.resize(mask, (1280, 720))
else:
    print("Warning: mask image not found. Continuing without mask.")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

counting_line = [180, 400, 673, 400]

counted_ids = set()
id_class_map = {}

vehicle_counts = {'car': 0, 'bus': 0, 'truck': 0}

while True:
    success, img = cap.read()
    if not success:
        break

    img_input = img.copy()
    if mask is not None:
        img_input = cv2.bitwise_and(img, mask)

    results = model(img_input, stream=True)

    detections = []  # Each item: [x1, y1, x2, y2, conf, class_name]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = classNames[cls]

            if class_name in ['car', 'bus', 'truck'] and conf > 0.5:
                detections.append([x1, y1, x2, y2, conf, class_name])
                cvzone.putTextRect(img, f"{class_name} {conf:.2f}", (max(0, x1), max(30, y1)),
                                   scale=0.8, thickness=1, offset=2)

# Prepare detections for SORT (only x1,y1,x2,y2,conf)
    dets_for_sort = np.array([d[:5] for d in detections]) if len(detections) > 0 else np.empty((0, 5))

    tracks = tracker.update(dets_for_sort)

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        tracked_box = [x1, y1, x2, y2]

        # Assign class to track ID via IoU matching
        if track_id not in id_class_map:
            max_iou = 0
            assigned_class = 'unknown'
            for det in detections:
                det_box = det[:4]
                det_class = det[5]
                iou = calculate_iou(tracked_box, det_box)
                if iou > max_iou:
                    max_iou = iou
                    assigned_class = det_class
            if max_iou >= 0.5:
                id_class_map[track_id] = assigned_class
            else:
                id_class_map[track_id] = 'unknown'

        class_name = id_class_map.get(track_id, 'unknown')

        bbox = (x1, y1, w, h)
        cvzone.cornerRect(img, bbox, colorR=(0, 255, 0), t=2)
        cvzone.putTextRect(img, f"{class_name} {track_id}", (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=3)

        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count when crossing line (with tolerance)
        if counting_line[0] < cx < counting_line[2] and counting_line[1] - 20 < cy < counting_line[1] + 20:
            if track_id not in counted_ids and class_name in vehicle_counts:
                counted_ids.add(track_id)
                vehicle_counts[class_name] += 1
                cv2.line(img, (counting_line[0], counting_line[1]), (counting_line[2], counting_line[3]), (0, 255, 0), 5)

    # Draw counting line red (default)
    cv2.line(img, (counting_line[0], counting_line[1]), (counting_line[2], counting_line[3]), (0, 0, 255), 5)

    # Show counts
    y_pos = 50
    for v_type, count in vehicle_counts.items():
        cvzone.putTextRect(img, f"{v_type}: {count}", (50, y_pos), scale=2, thickness=3, offset=10, colorR=(0, 0, 0))
        y_pos += 45

    cv2.imshow('Vehicle Counter', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinal counts:")
for v_type, count in vehicle_counts.items():
    print(f"{v_type}: {count}")
