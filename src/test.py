
import math
from pathlib import Path

import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

# from main import project_path
project_path = Path("main.py").parent.absolute().parent

model_path = project_path.joinpath("models/keras_model.h5")
labels_path = project_path.joinpath("models/labels.txt")
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)
classifier.data = np.ndarray(shape=(1, 224, 224), dtype=np.float32)

offset = 20
imgSize = 224

while True:
    _, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            _, result = classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            _, result = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, result, (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
