from dt_apriltags import Detector
import cv2
import numpy as np
import json

#retrieve camera intrinsic data from file
with open("info.json") as file:
    file_content = file.read()
info = json.loads(file_content)

#initialize tag detector
at_detector = Detector(
    families='tag16h5',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)
#length of tag's border (meters)
tag_size = 0.158

#camera intrinsic parameters (found through calibration)
camera_matrix = np.array(info["camera_matrix"])
dist = np.array(info["distortion"])
camera_params = (camera_matrix[0][0],camera_matrix[1][1],camera_matrix[0][2],camera_matrix[1][2])

#initialize camera
camera = cv2.VideoCapture(0)

#used to project world points (eg the corners of a tag) into image points (pixels)
def project(obj_point, camera_matrix, pose_R, pose_t):
    obj_point_4d = np.append(obj_point, [[1]], axis=0)
    camera_matrix_4by3 = np.append(camera_matrix,[[0],[0],[0]],axis=1)
    transform = np.append(pose_R,pose_t,axis=1)
    transform = np.append(transform,[[0,0,0,1]],axis=0)
    transform = np.dot(camera_matrix_4by3,transform)
    img_point = np.dot(transform, obj_point_4d)
    return img_point

while True:
    ret, color_img = camera.read()
    #image fed to detector must be grayscale
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    #find tags and weed out unlikely results (50 is an arbitrary limit to confidence)
    tags = at_detector.detect(img, True, camera_params, tag_size)
    tags = [tag for tag in tags if (tag.decision_margin > 50)]

    #points used for the demo - note that the world's origin is set at the center of the tag
    obj_points = [
        [[-tag_size/2],[tag_size/2],[-tag_size]],
        [[tag_size/2],[tag_size/2],[-tag_size]],
        [[tag_size/2],[-tag_size/2],[-tag_size]],
        [[-tag_size/2],[-tag_size/2],[-tag_size]]
    ]

    for tag in tags:
        #get the projected locations of the demo points
        img_points = [project(point, camera_matrix, tag.pose_R, tag.pose_t) for point in obj_points]
        img_points = [point * (1/point[2][0]) for point in img_points]

        #draw lines
        for idx in range(4):
            cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), thickness = 3)
            cv2.line(color_img, tuple(img_points[idx-1][:2].flatten().astype(int)), tuple(img_points[idx][:2].flatten().astype(int)), (0, 255, 0), thickness = 3)
            cv2.line(color_img, tuple(tag.corners[idx, :].astype(int)), tuple(img_points[idx][:2].flatten().astype(int)), (0, 255, 0), thickness = 3)

        #show tag's id
        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))

    cv2.imshow('Detected tags', color_img)

    cv2.waitKey(20)

