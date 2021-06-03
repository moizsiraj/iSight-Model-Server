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

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
# print(model)
model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 5),
                         nn.LogSoftmax(dim=1))
model.to(device);

state_dict = torch.load('5ClassesRevised-2.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

classes = ['bed', 'chair', "office chair", "sofa", 'table']

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def textOrObject(imagePath, model, classes):
    result = ""
    result = recogniseText(imagePath)
    print(len(result))
    if len(result) <= 25:
        result = classifyImage(imagePath, model, classes)
        return "This is a " + str(result)
    else:
        return result


def recogniseText(imagePath):
    # im = Image.open(imagePath)  # the second one
    # im = im.filter(ImageFilter.MedianFilter())
    # enhancer = ImageEnhance.Contrast(im)
    # im = enhancer.enhance(2)
    # im = im.convert('1')
    preprocessed = preprocessImage(imagePath)
    cv2.imwrite('preprocess.jpeg', preprocessed)
    # Image.open('preprocess.jpeg')
    text = pytesseract.image_to_string('preprocess.jpeg')
    return text


def classifyImage(path, model, classes):
    image = cv2.imread(path, 1)
    PILImage = Image.fromarray(image)
    testTransforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    input = testTransforms(PILImage)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)

    # output = model.forward(input[None, ...])

    output = model.forward(input[None])

    probabilityOutput = torch.exp(output)
    topProbability, predictedClass = probabilityOutput.topk(1, dim=1)
    predictedClass = torch.squeeze(predictedClass)

    mode = torch.mode(predictedClass, 0)
    # fig = plt.figure(figsize=(28, 8))
    # ax = fig.add_subplot(2, 20 / 2, 1, xticks=[], yticks=[])
    # plt.imshow(np.transpose(input.cpu().numpy(), (1, 2, 0)).astype('uint8'))
    # ax.set_title(classes[mode[0].item()])
    return classes[mode[0].item()]


@app.post('/filename')
def get_filename(fname: str):
    fileName = fname + '.jpg'
    print(fileName)
    path = "D:/Web Dev/Projects/FYP-Server/uploads/" + fileName
    result = textOrObject(path, model, classes)
    print(result)
    return result


@app.post('/filenameText')
def get_filename(fname: str):
    fileName = fname + '.jpg'
    print(fileName)
    path = "D:/Web Dev/Projects/FYP-Server/uploads/" + fileName
    text = recogniseText(path)
    print(text)
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


@app.get('/hello')
def get_hello():
    print("hi")
    return "hello"
