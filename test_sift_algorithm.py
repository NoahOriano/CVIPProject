import sift_algorithm
import cv2
import numpy as np

# Get the anchor image and generate keypoints and descriptors
anchor_image = cv2.imread('data\\Anchor\\20240925_064700703_iOS.png', cv2.IMREAD_COLOR)
# Convert the image to grayscale
anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
# Resize the image to a smaller size
anchor_image = cv2.resize(anchor_image, (0, 0), fx=0.25, fy=0.25)
kp, des = sift_algorithm.computeKeypointsAndDescriptors(anchor_image)

# Save the image with descriptors to file
image = cv2.drawKeypoints(anchor_image, kp, None)
cv2.imwrite('output\\anchor_image_keypoints.png', image)
