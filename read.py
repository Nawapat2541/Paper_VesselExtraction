import cv2 as cv

img = cv.imread('Pictures/training/images/21_training.tif')

cv.imshow('Retina1', img)

cv.waitKey(0)
