import cv2
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)

Y_error = 0
X_error = 0
area = 0

if not cap.isOpened():
    print("Cannot open camera or file")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    bgr = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr,cv2.COLOR_RGB2HSV)

    lower = np.array([0,80,150])
    upper = np.array([50,255,255])
    #mask the blue object
    mask_blue = cv2.inRange(hsv,lower,upper)

    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #draw contour
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 

    #print(approx)

    


    h,w,ch = frame.shape
    #print(h,w)

    #center of image's position
    x_centerImg = w // 2
    y_centerImg = h // 2

    #horizontal axis
    cv2.line(frame, (0, h//2), (w, h//2), (0,255,0),1)

    #vertical axis
    cv2.line(frame, (w//2, 0), (w//2, h), (0,255,0),1)
    
    if area < 25:   #set area threshold of 50 pixel squared
        X_error = 0
        Y_error = 0
    else: 
        # the co-ordinates of the vertices. 
        n = approx.ravel() # Used to flatted the array containing 
        # print(n)
        i = 0
        x = []
        y = []

        for j in n : 
            if(i % 2 == 0): 
                x.append(n[i]) 
                y.append(n[i + 1]) 
            i = i + 1
        
        # print("min x",min(x))
        #print("max x",max(x))
        #print("min y",min(y))
        #print("max y",max(y))

        #draw bounding box
        cv2.rectangle(frame, (min(x), min(y)), (max(x),max(y)), (0, 255, 0), 2)
        
        #center of the bounding box's position
        x_centerBox = (min(x) + max(x)) // 2
        y_centerBox = (min(y) + max(y)) // 2
        point = [x_centerBox, y_centerBox]
        cv2.circle(frame, point, 4, (0, 255, 0), -1)
        
        #find positional errors
        X_error = x_centerBox -  x_centerImg
        Y_error = -(y_centerBox - y_centerImg)

        #draw line from center of box to center of image
        cv2.arrowedLine(frame, (x_centerImg, y_centerImg),(x_centerBox, y_centerBox) , (0,0,255),1)
        
    #show distance error in x,y axes
    cv2.putText(frame, f"X_error: {X_error} px", (w - 300, h - 100), font, 0.8, [0,255,0], 2)
    cv2.putText(frame, f"Y_error: {Y_error} px", (w - 300, h - 50), font, 0.8, [0,255,0], 2)    

    cv2.imshow("image",frame)



    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()