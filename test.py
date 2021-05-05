from os import listdir
import cv2
from fastapi import FastAPI, File, UploadFile
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torch
from torch import nn
import pytesseract
import time
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models


def recogniseText(imagePath):
    # im = Image.open(imagePath)  # the second one
    # im = im.filter(ImageFilter.MedianFilter())
    # enhancer = ImageEnhance.Contrast(im)
    # im = enhancer.enhance(2)
    # im = im.convert('1')
    preprocessed = preprocessImage(imagePath)
    cv2.imwrite('preprocess.jpeg', preprocessed)
    # Window name in which image is displayed
    window_name = 'image'

    # Using cv2.imshow() method
    # Displaying the image
    cv2.imshow(window_name, preprocessed)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
    # Image.open('preprocess.jpeg')
    text = pytesseract.image_to_string('preprocess.jpeg')
    return text


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return


def preprocessImage(path):
    print(path)
    image = cv2.imread(path)
    grayImg = get_grayscale(image)
    threshImg = thresholding(grayImg)
    # openingImg = opening(threshImg)
    ##cannyImg = canny(openingImg)
    return threshImg


recogniseText("C:/Users/moizs/OneDrive/Desktop/test.jpeg")
