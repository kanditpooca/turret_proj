import cv2
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO("turret_proj/human.pt")

width = 640
height = 640

bb = [0,0,0,0]
sum = 0
font = cv2.FONT_HERSHEY_DUPLEX

class_list = ['human']

detection_colors = [(0,255,0)]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera or file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # resize the frame | small frame optimise the run
    frame = cv2.resize(frame, (width, height))

    # Run YOLOv8 inference on the frame
    detect_params = model.predict(source=frame,save=False,conf=0.6, verbose=False)
    # print(detect_params[0])
    # Convert tensor array to numpy
    DP = detect_params[0].cuda()
    DP = DP.cpu()
    DP = DP.to('cpu')
    DP = DP.numpy()

    if (len(DP) != 0):
        for i in range(len(detect_params[0])):

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls[0].cuda()
            clsID = clsID.cpu()
            clsID = clsID.to('cpu')
            clsID = clsID.numpy()
            
            conf = box.conf[0].cuda()
            conf = conf.cpu()
            conf = conf.to('cpu')
            conf = conf.numpy()
            
            bb = box.xyxy[0].cuda()
            bb = bb.cpu()
            bb = bb.to('cpu')
            bb = bb.numpy()

            all_cls = boxes.cls.cuda()
            all_cls = all_cls.cpu()
            all_cls = all_cls.to('cpu')
            all_cls = all_cls.numpy()

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(np.round(conf*100, 1)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                detection_colors[int(clsID)],
                2,
            )
    
    h,w,ch = frame.shape
    #print(h,w)

    #center of image's position
    x_centerImg = w // 2
    y_centerImg = h // 2

    #horizontal axis
    cv2.line(frame, (0, h//2), (w, h//2), (0,255,0),1)

    #vertical axis
    cv2.line(frame, (w//2, 0), (w//2, h), (0,255,0),1)

    #center of the bounding box's position
    x_centerBox = (int(bb[0]) + int(bb[2])) // 2
    y_centerBox = (int(bb[1]) + int(bb[3])) // 2
    point = [x_centerBox, y_centerBox]

    #draw line from center of box to center of image and center point
    if len(detect_params[0]) > 0:
        cv2.arrowedLine(frame, (x_centerImg, y_centerImg),(x_centerBox, y_centerBox) , (0,0,255),1)
        cv2.circle(frame, point, 4, (0, 255, 0), -1)

    #show distance error in x,y axes
    X_error = x_centerBox -  x_centerImg
    Y_error = -(y_centerBox - y_centerImg)
    if len(detect_params[0]) == 0:
        X_error = 0
        Y_error = 0
    cv2.putText(frame, f"X_error: {X_error} px", (w - 300, h - 100), font, 0.8, [0,255,0], 2)
    cv2.putText(frame, f"Y_error: {Y_error} px", (w - 300, h - 50), font, 0.8, [0,255,0], 2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

    # Display the resulting frame
    sum = 0
    cv2.imshow("1", frame)

cap.release()
cv2.destroyAllWindows()