#tests the camera intrinsic parameters found through calibration - all lines in the video feed should be straight

import cv2
import numpy
import json

with open("info.json") as file:
    file_content = file.read()
info = json.loads(file_content)

mtx = numpy.array(info["camera_matrix"])
dist = numpy.array(info["distortion"])

camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('calibresult.png', dst)
    cv2.waitKey(20)

