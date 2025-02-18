import cv2
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX

path = "blue_test1.jpg"
img = cv2.imread(path)
bgr = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr,cv2.COLOR_RGB2HSV)

lower = np.array([90,78,94])
upper = np.array([124,255,255])
#mask the blue object
mask_blue = cv2.inRange(hsv,lower,upper)

contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#draw contour
#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 

#print(approx)

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

print("min x",min(x))
print("max x",max(x))
print("min y",min(y))
print("max y",max(y))

h,w,ch = img.shape
print(h,w)

#center of image's position
x_centerImg = w // 2
y_centerImg = h // 2

#horizontal axis
cv2.line(img, (0, h//2), (w, h//2), (0,255,0),1)

#vertical axis
cv2.line(img, (w//2, 0), (w//2, h), (0,255,0),1)

#draw bounding box
cv2.rectangle(img, (min(x), min(y)), (max(x),max(y)), (0, 255, 0), 2)

#center of the bounding box's position
x_centerBox = (min(x) + max(x)) // 2
y_centerBox = (min(y) + max(y)) // 2
point = [x_centerBox, y_centerBox]
cv2.circle(img, point, 4, (0, 255, 0), -1)

#draw line from center of box to center of image
cv2.arrowedLine(img, (x_centerImg, y_centerImg),(x_centerBox, y_centerBox) , (0,0,255),1)

#show distance error in x,y axes
X_error = x_centerBox -  x_centerImg
Y_error = -(y_centerBox - y_centerImg)
cv2.putText(img, f"X_error: {X_error} px", (w - 300, h - 100), font, 0.8, [0,255,0], 2)
cv2.putText(img, f"Y_error: {Y_error} px", (w - 300, h - 50), font, 0.8, [0,255,0], 2)


cv2.imshow("image",img)



if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()