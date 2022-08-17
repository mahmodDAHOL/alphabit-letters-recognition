from pathlib import Path

import cv2
import numpy as np


# this function take path for image and return descriptors list of that image
def extract_sift(path: Path):
    frame = cv2.imread(str(path))

    frame = cv2.resize(frame, (128, 128))

    # convert image from RGB to gray  (8 bit for every pixel)
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # convert image from RGB to HSV  (HSV is a color model that
    #                                less sensitive to shadow in the image)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lowerBoundary and upperBoundary are values that have been experimented by the author
    # to determine  H, S and V (HSV values) to mask the object that we intrested in
    # (the hand in this project)
    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")

    # after determine the boundary we will use inRange method from opencv library
    # for applying these values to mask the hand
    # the mask mean convert the hand to white and background to black
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    # addWeight method is for overlaying one image over another image
    # see https://www.geeksforgeeks.org/opencv-alpha-blending-and-masking-of-images/
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)

    # medianBlur method for applying filter with size 5*5 on image to eliminating the noise
    skinMask = cv2.medianBlur(skinMask, 5)

    # this line for apply the mask that we created for make the hand white, background black
    skin = cv2.bitwise_and(converted2, converted2, mask=skinMask)

    # Canny method for edge detection and 60 is hyperparameter
    # see https://www.geeksforgeeks.org/python-opencv-canny-function/
    img2 = cv2.Canny(skin, 60, 60)

    # initializing sift algorithm that extract descriptors that we will use them for training
    # classification model
    sift = cv2.xfeatures2d.SIFT_create()
    # resizing
    img2 = cv2.resize(img2, (256, 256))

    # useing detectAndCompute method for apply the algorithm on the image
    # it return des variable (descriptors) list of points that describe the feature of image
    # and kp variable (keypoits) that will show on image as points
    # for more info see https://www.youtube.com/watch?v=DZtUt4bKtmY
    kp, des = sift.detectAndCompute(img2, None)

    # drawKeypoints method for draw the points (des) on the image
    img2 = cv2.drawKeypoints(img2, kp, None, (0, 0, 255), 4)

    return des

def save_training_results(content: str, path_to_save: Path):
    filepath = path_to_save.joinpath("result.txt")
    with filepath.open("a", encoding="utf-8") as file:
        file.write(f"{content}\n\n\n")
