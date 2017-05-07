import glob
import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt


class CalibrationImages(object):
    def __init__(self):
        #Prepare Object Points like (0,0,0),(1,0,0), (2,0,0) ...(8,5,0)
        self.objpoints = [] #3D points in real world space. Z co-ordinates will all be zero in the 2D Plane
        self.imgpoints = [] #2D points in image plane


    def __readImages__(self,path):
        return glob.glob(path)

    def __cal_setObjectImagePoints__(self, corners, objp):
        self.imgpoints.append(corners)
        self.objpoints.append(objp)

    def cal_getObjectImagePoints(self):
        return self.objpoints,self.imgpoints

    def calibrateAndDisplayImages(self,path):

        images = self.__readImages__(path)
        objp = np.zeros((6*9,3),np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) #x, y coordinates

        for fname in images:
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
            if ret == True:
                self.__cal_setObjectImagePoints__(corners,objp)
                img = cv2.drawChessboardCorners(img,(9,6),corners,ret)
                plt.figure(figsize=(12,12))
                plt.imshow(img)
                plt.show()

    def cal_undistort(self,img):
        # Use cv2.calibrateCamera() and cv2.undistort()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img.shape[0:2], None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        return dst

    def cal_undistort_for_all(self,path):
        images = self.__readImages__(path)
        undistorted_images = []
        for test_img in images:
            img = mpimg.imread(test_img)
            undistorted = self.cal_undistort(img)
            undistorted_images.append(undistorted)
        return images, undistorted_images

    def getBinaryWarped(self,image):
        img_size = (image.shape[1], image.shape[0])
        width, height = img_size

        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
            [((img_size[0] / 6) - 10), img_size[1]],
            [(img_size[0] * 5 / 6) + 60, img_size[1]],
            [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        dst = np.float32(
            [[(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0]])
        M = cv2.getPerspectiveTransform(src,dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        binary_warped = cv2.warpPerspective(image,M, (width, height))
        return binary_warped,M,Minv


