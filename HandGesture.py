import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

width, height = 1280, 720
folderPath = "/Users/carliapriansyahh/Downloads/pythonProject/PowerPoint"
gestureThreshold = 300
buttonDelay = 10

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
pathImages = sorted(os.listdir(folderPath))
imgNumber = 0
hs, ws = int(120 * 1), int(213 * 1)
buttonPressed = False
buttonCounter = 0
annotations = [[]]
annotationNumber = 0
annotationStart = False
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if imgNumber < len(pathImages):
        pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
        imgCurrent = cv2.imread(pathFullImage)

        if imgCurrent is None:
            print(f"Failed to load image at {pathFullImage}. Skipping to next image.")
            if imgNumber < len(pathImages) - 1:
                imgNumber += 1
            continue
    else:
        print("No more images in folder.")
        break

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and not buttonPressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand["center"]
        lmList = hand["lmList"]

        xVal = int(np.interp(lmList[8][0], [0, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [0, height], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:
            annotationStart = False

            if fingers == [0, 1, 0, 0, 1]:
                print("Left")
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber -= 1

            elif fingers == [1, 1, 0, 0, 0]:
                print("Right")
                if imgNumber < len(pathImages) - 1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber += 1

        elif fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationStart = False

        elif fingers == [0, 1, 0, 0, 0]:
            if not annotationStart:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)

        elif fingers == [1, 1, 1, 1, 1]:
            if annotations:
                annotations = [[]]
                annotationNumber = 0
                buttonPressed = True

    else:
        annotationStart = False

    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonPressed = False
            buttonCounter = 0

    for annotation in annotations:
        for j in range(1, len(annotation)):
            cv2.line(imgCurrent, annotation[j - 1], annotation[j], (255, 0, 255), 12)

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:w] = imgSmall

    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()