import numpy as np
import cv2

class Threshold(object):
    def __init__(self):
        self.ksize = 3 # Choose a larger odd number to smooth gradient measurements

    def abs_sobel_thresh(self,img, orient='x', sobel_kernel=3, thresh=(0,255)):
        
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output

    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(0, 255)):
        
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Define a function to threshold an image for a given range and Sobel kernel
    def dir_threshold(self,img, sobel_kernel=3, thresh=(0, np.pi/2)):
        
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def gen_threshold_combined(self,image):
        # convert to gray scale
        img = np.copy(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        height, width = gray.shape
        
        # apply gradient threshold on the horizontal gradient
        sx_binary = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=self.ksize, thresh=(20, 100))

        # apply gradient direction threshold so that only edges closer to vertical are detected.
        dir_binary = self.dir_threshold(gray, sobel_kernel=self.ksize, thresh=(np.pi/6, np.pi/2))

        # combine the gradient and direction thresholds.
        combined_condition = ((sx_binary == 1) & (dir_binary == 1))

        color_threshold = 150
        R = img[:,:,0]
        G = img[:,:,1]
        color_combined = np.zeros_like(R)
        r_g_condition = (R > color_threshold) & (G > color_threshold)

        # color channel thresholds
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        S = hls[:,:,2]
        L = hls[:,:,1]

        # S channel performs well for detecting bright yellow and white lanes
        s_thresh = (100, 255)
        s_condition = (S > s_thresh[0]) & (S <= s_thresh[1])

        l_thresh = (120, 255)
        l_condition = (L > l_thresh[0]) & (L <= l_thresh[1])

        color_combined[(r_g_condition & l_condition) & (s_condition | combined_condition)] = 1

        return color_combined


