
import numpy as np
import cv2 as cv
import glob

#chess = (31, 23)
chess = (12, 13)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria that will end after 30 iterations or when pixel treshhold 0.001 is met

objp = np.zeros((chess[0]*chess[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chess[0],0:chess[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = sorted(glob.glob("camera_calib/*.jpg"))  # glob for processing each image in the loop
images = sorted(glob.glob("tif/*.tif"))
if not images:
    print("0 imgaes load wrong path while loading file :/")
    exit()

img_gray = None

for image in images:
    img_color = cv.imread(image)
    if img_color is None:
        print(f"couldnot load the {image}")
        continue

    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)   # converts all images to grayscale for faster detection
    flags = (cv.CALIB_CB_ADAPTIVE_THRESH |
             cv.CALIB_CB_NORMALIZE_IMAGE |
             cv.CALIB_CB_FAST_CHECK)
    status, corners = cv.findChessboardCorners(img_gray, chess, flags) # detect cornners if detected it add object points an store 2d image

    if status:
        corners2 = cv.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img_color, chess, corners2, True)                                  # this condition diplay each corner segment brifly for user verification
        cv.imshow('img', img_color)
        cv.waitKey(700)

cv.destroyAllWindows()

if img_gray is not None and len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_gray.shape[::-1], None, None)

#cv.calibrateCamera(??) # camera calibration returns the camera matrix, distortion coefficients, rotation and translation vectors
print(ret) # reprojection error
print(mtx) # camera matrix
print(dist) # distortion
print(rvecs) # rot vect
print(tvecs) # trans vector
cv.destroyAllWindows()