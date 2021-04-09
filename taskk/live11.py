from cv2 import cv2
import numpy as np
import imutils
from functools import reduce
import operator
import math

################ Capture photo from live cam ############
# cam = cv2.VideoCapture(0)

# cv2.namedWindow("test")

# img_counter = 0

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print("failed to grab frame")
#         break
#     cv2.imshow("test", frame)

#     k = cv2.waitKey(1)
#     if k%256 == 27:# ESC 
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32: # SPACE 
#         img_name = "opencv_frame_{}.png".format(img_counter)
#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#         img_counter += 1

# cam.release()

# cv2.destroyAllWindows()
##################################################################################################333
frame = cv2.imread('1.jpg')  ##################test###############
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)
#ray = cv2.GaussianBlur(gray, (11, 11), 0)
ret,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
cv2.imshow('thresh', thresh)
outerBox = cv2.bitwise_not(thresh)
cv2.imshow('BitwiseNot', outerBox)


contours = cv2.findContours(outerBox.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an output of all zeroes that has the same shape as the input
# image
out = np.zeros_like(frame)
cnts = imutils.grab_contours(contours)
c = max(cnts, key=cv2.contourArea)

cv2.imshow('Donut', frame) 


# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])
print(extLeft, extRight, extTop,extBot)
cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
cv2.circle(frame, extTop, 8, (255, 0, 0), -1)
cv2.circle(frame, extBot, 8, (255, 255, 0), -1)

coords = [extLeft, extRight, extTop, extBot]
center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
sortedlist=sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
print(sortedlist)



pt1= np.float32([sortedlist])
height, width = 300,400
#pt2= np.float32([[0,height],[0,0],[width,0],[width,height]])
pt2=np.float32([[0,0],[0,height],[width,height],[width,0]])


materix = cv2.getPerspectiveTransform(pt1,pt2)
output = cv2.warpPerspective(frame, materix, (width,height))

cv2.imshow("output", output)


cv2.waitKey(0)
cv2.destroyAllWindows()




