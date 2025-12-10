import numpy as np
import cv2 as cv
import glob

#chess = (31,23)
chess = (12,13)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chess[0]*chess[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chess[0],0:chess[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
#images = glob.glob("camera_calib/*.jpg")
images = glob.glob("tif/*.tif")
img_gray = None
for name in images:
    img_color = cv.imread(name)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)

    status, corners = cv.findChessboardCorners(img_gray, chess, None)

    if status == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img_color, chess, corners2, status)
        cv.imshow('img', img_color)
        cv.waitKey(1000)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)
print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)
cv.destroyAllWindows()