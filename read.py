import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Pictures/training/images/21_training.tif')
resized_img = cv.resize(img, dsize=(584, 565))
height, width, channels = resized_img.shape
print(height, width, channels)
cv.imshow('input_img', img)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray_img', gray_img)

(thresh, bw_img) = cv.threshold(gray_img, 130, 255, cv.THRESH_BINARY)
print(thresh)
cv.imshow('binary_img', bw_img)

denoted = (1, 7)
print(denoted)
kernel = np.ones(denoted, np.uint8)
# kernel[0, 0] = 0
# kernel[0, 1] = 0
# kernel[0, 3] = 0
# kernel[0, 4] = 0
# kernel[1, 0] = 0
# kernel[1, 4] = 0
# kernel[3, 0] = 0
# kernel[3, 4] = 0
# kernel[4, 0] = 0
# kernel[4, 1] = 0
# kernel[4, 3] = 0
# kernel[4, 4] = 0
print(kernel)

# dilation_img = cv.dilate(bw_img, kernel, iterations=1)
# cv.imshow('dilation_img', dilation_img)
#
# erosion_img = cv.erode(bw_img, kernel, iterations=1)
# cv.imshow('erosion_img', erosion_img)
#
# opening_step_img = cv.dilate(erosion_img, kernel, iterations=1)
# cv.imshow('opening_step_img', opening_step_img)

opening_img = cv.morphologyEx(bw_img, cv.MORPH_OPEN, kernel)
cv.imshow('opening_img', opening_img)

# closing_step_img = cv.erode(dilation_img, kernel, iterations=1)
# cv.imshow('closing_step_img', closing_step_img)
#
# closing_img = cv.morphologyEx(bw_img, cv.MORPH_CLOSE, kernel)
# cv.imshow('closing_img', closing_img)

# top_step_hat_img = bw_img
# for i in range(height):
#     for j in range(width):
#         if opening_step_img[i, j] == 255:
#             top_step_hat_img[i, j] = 0
#
# cv.imshow('top_step_hat_img', top_step_hat_img)

top_hat_img = cv.morphologyEx(bw_img, cv.MORPH_TOPHAT, kernel, iterations=1)
cv.imshow('top_hat_img', top_hat_img)

# gradient_img = cv.morphologyEx(bw_img, cv.MORPH_GRADIENT, kernel, iterations=1)
# cv.imshow('gradient_img', gradient_img)

# black_step_hat_img = gray_img
# for i in range(height):
#     for j in range(width):
#     for j in range(width):
#         if closing_step_img[i, j] == 255:
#             black_step_hat_img[i, j] = 0
#
# cv.imshow('black_step_hat_img', black_step_hat_img)

# black_hat_img = cv.morphologyEx(bw_img, cv.MORPH_BLACKHAT, kernel)
# cv.imshow('black_hat_img', black_hat_img)

ga_img = cv.GaussianBlur(top_hat_img, (5, 5), 0)
cv.imshow('Gassian_filter_img', ga_img)

cv.waitKey(0)
