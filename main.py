import cv2 
import numpy as np

print("OpenCV version:")
print(cv2.__version__)


image = cv2.imread("match.jpg")

#lines = cv2.HoughLines(edges, 1, np.pi/180, 200)


m = np.array([[110,0], [110, 55], [293.2, 55], [293.2, 0]]) #fields real coordinates in meters
px = np.array([[303, 117], [180, 131], [430, 227], [566, 206]]) #pixels coodinates

homo, mask = cv2.findHomography(px, m, cv2.RANSAC, 5.0)

print(homo)

h, w, c = image.shape

img_homo = cv2.warpPerspective(image, homo, (w, h))

cv2.line(img_homo, (293, 55), (115, 55), (0,255,0), 2)

#homo, mask = cv2.findHomography(m, px, cv2.RANSAC, 5.0)
homo_inverse = np.linalg.inv(homo)
img_straight = cv2.warpPerspective(img_homo, homo_inverse, (w, h))


#Segmentation and threshholding
'''
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

#Define a mask ranging from lower to uppper
mask = cv2.inRange(hsv, lower_green, upper_green)
 #Do masking
res = cv2.bitwise_and(image, image, mask=mask)
 #convert to hsv to gray
res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
'''

#Defining a kernel to do morphological operation in threshold #image to get better output.
'''
kernel = np.ones((13,13),np.uint8)
thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV |  cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
'''
cv2.imshow("BGR", image)
#cv2.imshow("BGR", res_bgr)
#cv2.imshow("Gray", res_gray)
cv2.imshow("Homo", img_homo)
cv2.imshow("Straight", img_straight)

cv2.waitKey(0)
cv2.destroyAllWindows()
