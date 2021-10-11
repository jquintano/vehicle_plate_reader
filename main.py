import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #captureDevice = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

def preprocess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    imgDilate = cv2.dilate(imgCanny, kernel=np.ones((3,3)), iterations=2)
    imgProcessed = cv2.erode(imgDilate, kernel=np.ones((3,3)), iterations=1)
    return imgProcessed

def reorder(pts):
    pts = pts.reshape((4, 2))
    ptsNew = np.zeros((4, 1, 2), np.int32)
    add = pts.sum(1)
    ptsNew[0] = pts[np.argmin(add)]
    ptsNew[3] = pts[np.argmax(add)]
    diff = np.diff(pts, axis=1)
    ptsNew[1] = pts[np.argmin(diff)]
    ptsNew[2] = pts[np.argmax(diff)]
    return  ptsNew

def getWarp(img, biggest):
    biggest = reorder(biggestit)
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgOut = cv2.warpPerspective(img, matrix, (frameWidth, frameHeight))
    return imgOut

def getContour(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            # cv2.drawContours(imgContour, cnt, -1, (255,0,0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print(area)
            print(len(approx))
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

while True:
    _, img = cap.read()
    cv2.resize(img,(640, 480))
    imgContour = img.copy()

    imgProcessed = preprocess(img)
    biggest = getContour(imgProcessed)
    print(biggest)
    imgWarp = getWarp(img, biggest)
    # cv2.imshow("CAM", img)


    cv2.imshow("RESULT", imgWarp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break