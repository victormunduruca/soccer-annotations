import cv2 
import numpy as np
import math
from cvtools import hough

#Read image file
image = cv2.imread("match2.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 35, 35])
upper_green = np.array([70, 255, 255])

#Define a mask ranging from lower to uppper
mask = cv2.inRange(hsv, lower_green, upper_green)
#Do masking
res = cv2.bitwise_and(image, image, mask=mask)
 #convert to hsv to gray
res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)


#Defining a kernel to do morphological operation in threshold #image to get better output.
kernel = np.ones((13,13),np.uint8)
thresh = cv2.threshold(res_gray,100,255,cv2.THRESH_BINARY_INV |  cv2.THRESH_OTSU)[1]
#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.bitwise_not(thresh)

res_mask = cv2.bitwise_and(image, image, mask=thresh)

#cv2.imshow("Segmented image", res_mask)


#Hough line tests
gray = cv2.cvtColor(res_mask, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,110,150,apertureSize = 3)
dilate_kernel = np.ones((2,2),np.uint8)
dilation = cv2.dilate(edges,dilate_kernel,iterations = 1)
cv2.imshow('edges', dilation)
lines = cv2.HoughLinesP(dilation,1,np.pi/180,100,minLineLength=200,maxLineGap=10)

'''
rho, theta, thresh = 2, np.pi/180, 400
lines = cv2.HoughLines(edges, rho, theta, thresh)
    
segmented = hough.segment_angle_kmeans(lines)
img_lines = hough.draw_lines(edges, lines, (0, 255, 0))

cv2.imshow('Image with lines', img_lines)
print(segmented)
#intersections = segmented_intersections(segmented)

#print(intersections)
'''
#for point in intersections:
    #cv2.circle(edges, point, 1, (0, 255, 0), thickness=1, lineType=8, shift=0)
  #  print(point)


for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(res_mask,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("Lines", res_mask)


cv2.waitKey(0)
cv2.destroyAllWindows()


