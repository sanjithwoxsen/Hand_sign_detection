import math
from cvzone.ClassificationModule import Classifier
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
counter = 0
offset = 20
imgSize = 244


labels = ["A","B","C","1","2","7","Thumbs-Up"]


while True:
    success, img = cap.read()
    imgoutput = img.copy()
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
            wcal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wcal, imgSize))
            imgResizeshape = imgResize.shape
            wGap = math.ceil((imgSize-wcal)/2)
            imgwhite[:, wGap:wcal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgwhite)
            print(prediction,index)
        else:
            k = imgSize / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hcal))
            imgResizeshape = imgResize.shape
            hGap = math.ceil((imgSize - hcal) / 2)
            imgwhite[hGap:hcal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgwhite)

        cv2.putText(imgoutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("Imagewhite", imgwhite)
    cv2.imshow("Image", imgoutput)
    cv2.waitKey(1)






