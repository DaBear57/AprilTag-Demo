import numpy as np
import cv2
import keyboard as kb
import json

#criteria for more precise corner points
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#world points of the corners of the chessboard
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

camera = cv2.VideoCapture(0)

kb.add_hotkey('c', lambda: append_corners())

#append one instance of found chessboard to list of points
def append_corners():
    try:
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        print(len(objpoints))
    except:
        print("chessboard not found")

while len(objpoints) < 5:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = []

    #show chessboard corners if found when space is held
    if kb.is_pressed(" "):
        try:
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            corners2 = cv2.cornerSubPix(gray, np.array(corners), (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(img, (9,6), corners2, True)
        except:
            pass

    cv2.imshow('img',img)
    cv2.waitKey(20)

#write to json file
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx, dist)
dictionary = {
    "camera_matrix" : [list(i) for i in mtx],
    "distortion" : list(dist[0])
}
print(dictionary)

with open("info.json","w") as file:
    json.dump(dictionary, file)