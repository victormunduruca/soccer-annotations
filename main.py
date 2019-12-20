import cv2 
import numpy as np
import math

def convert_coordinate(x, y, homography):
    p = np.array((x,y,1)).reshape((3,1))
    temp_p = homography.dot(p)
    sum = np.sum(temp_p ,1)
    px = int(round(sum[0]/sum[2]))
    py = int(round(sum[1]/sum[2]))
    return px, py

def magnitude(a, b):
    x_d = b[0]-a[0]
    y_d = b[1] - a[1]
    mag = math.sqrt((x_d)**2 + (y_d)**2)
    return mag
    
#Read image file
image = cv2.imread("match.jpg")

#lines = cv2.HoughLines(edges, 1, np.pi/180, 200)


#Fields real coordinates in meters, manually input
m = np.array([[110,0], [110, 55], [293.2, 55], [293.2, 0]])

#pixels coodinates
px = np.array([[303, 117], [180, 131], [430, 227], [566, 206]])

#Homography between coodinates in pixels and field dimensions
homo, mask = cv2.findHomography(px, m, cv2.RANSAC, 5.0)

px, py = convert_coordinate(180, 131, homo)

a = np.array([303, 117])
b = np.array([566, 206])

ax, ay = convert_coordinate(a[0], a[1], homo)
bx, by = convert_coordinate(b[0], b[1], homo)

a_np = np.array([ax, ay])
b_np = np.array([bx, by])


print(magnitude(a_np, b_np))

'''
#image dimensions
h, w, c = image.shape

# Create a black image
img_blk = np.zeros((512,512,3), np.uint8)

# Black imaged on the perspective of the field dimensions (top-view)
img_homo = cv2.warpPerspective(img_blk, homo, (w, h))

#Draw line on transformed black image
cv2.line(img_homo, (293, 30), (115, 30), (255,255,255), 2)

#Inverse of the calculated homography
homo_inverse = np.linalg.inv(homo)

#Black image with line restored to the original perspective
img_straight = cv2.warpPerspective(img_homo, homo_inverse, (w, h))

cv2.imshow("Black image with line", img_straight)


#Segmentation and threshholding

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

res_line = cv2.bitwise_and(img_straight, img_straight, mask=thresh)

cv2.imshow("Original image", res_line)

img_add = cv2.add(res_line, image)

cv2.imshow("Image with line", img_add)

#cv2.imshow("Original image", image)
#cv2.imshow("Image with masked line", res_gray)
#cv2.imshow("BW image with threshhold", thresh)


#cv2.imshow("Gray", res_gray)
#cv2.imshow("Homo", img_homo)
#cv2.imshow("Straight", img_straight)

'''


cv2.waitKey(0)
cv2.destroyAllWindows()
