import cv2 
import numpy as np

print("OpenCV version:")
print(cv2.__version__)


image = cv2.imread("match.jpg")

#lines = cv2.HoughLines(edges, 1, np.pi/180, 200)


#fields real coordinates in meters
m = np.array([[110,0], [110, 55], [293.2, 55], [293.2, 0]])

#pixels coodinates
px = np.array([[303, 117], [180, 131], [430, 227], [566, 206]])

#Homography between coodinates in pixels and field dimensions
homo, mask = cv2.findHomography(px, m, cv2.RANSAC, 5.0)

#image dimensions
h, w, c = image.shape

# Create a black image
img_blk = np.zeros((512,512,3), np.uint8)

# Black imaged on the perspective of the field dimensions (top-view)
img_homo = cv2.warpPerspective(img_blk, homo, (w, h))

#Draw line on transformed black image
cv2.line(img_homo, (293, 45), (115, 45), (255,255,255), 2)

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

#cv2.imshow("Original image", image)
#cv2.imshow("Image with masked line", res_gray)
#cv2.imshow("BW image with threshhold", thresh)


#cv2.imshow("Gray", res_gray)
#cv2.imshow("Homo", img_homo)
#cv2.imshow("Straight", img_straight)

cv2.waitKey(0)
cv2.destroyAllWindows()
