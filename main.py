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

model = models.resnet18(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
# print(model)
model.fc = nn.Sequential(nn.Linear(512, 128),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(128, 2),
                         nn.LogSoftmax(dim=1))
model.to(device);

state_dict = torch.load('chairdoorLatest.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

classes = ['Chair', 'Door']

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def recogniseText(imagePath):
    im = Image.open(imagePath)  # the second one
    im = im.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('1')
    im.save('filteredText.jpeg')
    text = pytesseract.image_to_string(Image.open('filteredText.jpeg'))
    return text


def classifyImage(image, model, classes):
    testTransforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    input = testTransforms(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)

    # output = model.forward(input[None, ...])

    output = model.forward(input[None])

    probabilityOutput = torch.exp(output)
    topProbability, predictedClass = probabilityOutput.topk(1, dim=1)
    predictedClass = torch.squeeze(predictedClass)

    mode = torch.mode(predictedClass, 0)
    fig = plt.figure(figsize=(28, 8))
    ax = fig.add_subplot(2, 20 / 2, 1, xticks=[], yticks=[])
    plt.imshow(np.transpose(input.cpu().numpy(), (1, 2, 0)).astype('uint8'))
    ax.set_title(classes[mode[0].item()])
    return classes[mode[0].item()]


@app.post('/filename')
def get_filename(fname: str):
    fileName = fname + '.jpg'
    print(fileName)
    path = "D:/Resumes/Dev-Reddit/uploads/" + fileName
    image = cv2.imread(path, 1)
    PILImage = Image.fromarray(image)
    print(classifyImage(PILImage, model, classes))
    return fname + '.jpg'


@app.post('/filenameText')
def get_filename(fname: str):
    fileName = fname + '.jpg'
    print(fileName)
    path = "D:/Resumes/Dev-Reddit/uploads/" + fileName
    text = recogniseText(path)
    print(text)

    return fname + '.jpg'
