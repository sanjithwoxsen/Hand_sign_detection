import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
counter = 0
offset = 20
imgSize = 500

folder = "Data/A"


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imgwhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectratio = h/w

        if aspectratio > 1:
            k = imgSize/h
            wcal =math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wcal, imgSize))
            imgResizeshape = imgResize.shape
            wGap = math.ceil((imgSize-wcal)/2)
            imgwhite[:, wGap:wcal+wGap] = imgResize
        else:
            k = imgSize / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hcal))
            imgResizeshape = imgResize.shape
            hGap = math.ceil((imgSize - hcal) / 2)
            imgwhite[hGap:hcal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgwhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(counter)



