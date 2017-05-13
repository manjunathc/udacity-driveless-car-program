import numpy as np
import cv2
import pickle
from utils.Smoothening import Smoothening 
from utils.FeatureExtractorUtil import FeatureExtractorUtil
from scipy.ndimage.measurements import label

# Applying the classifier for image frames
class ApplyClassifier(object):
    def __init__(self):
        self.smoothening = Smoothening()
        self.featureExtractorUtil = FeatureExtractorUtil()
        self.scale = 1.5
        self.ystart = 400
        self.ystop = 700
        # Read from Dump file for all values
        dist_pickle = pickle.load( open("model-params.pk", "rb" ) )
        self.svc = dist_pickle["svc"]
        print("svc-->",self.svc)
        self.X_scaler = dist_pickle["X_scaler"]
        self.color_space = dist_pickle["color_space"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.hist_bins = dist_pickle["hist_bins"]

    
    # Applying the classifier in a image frame
    # Define a single function that can extract features using hog sub-sampling and make predictions
    #def find_cars(self, img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,withboxes=False):
    def find_cars(self, img, withboxes=False):
        scale = self.scale
        ystart = self.ystart
        ystop = self.ystop
        pix_per_cell = self.pix_per_cell
        cell_per_block = self.cell_per_block
        spatial_size = self.spatial_size
        hist_bins = self.hist_bins
        orient = self.orient
        draw_img = np.copy(img)
        copied_image = np.copy(img)
        img = img.astype(np.float32)/255
        boxes = []

        img_tosearch = img[ystart:ystop,:,:]
        conv='RGB2YCrCb'
        ctrans_tosearch = self.smoothening.convert_color(img_tosearch, conv)
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = self.orient*self.cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        feature_vec=False
        vis = False
        hog1 = self.featureExtractorUtil.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis, feature_vec)
        hog2 = self.featureExtractorUtil.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis, feature_vec)
        hog3 = self.featureExtractorUtil.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis, feature_vec)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                subimg = ctrans_tosearch[ytop:ytop + window, xleft:xleft + window]

                # Get color features
                size=self.spatial_size
                nbins=self.hist_bins
                spatial_features = self.featureExtractorUtil.bin_spatial(subimg, size)
                hist_features = self.featureExtractorUtil.color_hist(subimg, nbins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    box = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                    #print("box-->",box)
                    #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                    if (withboxes == True):
                    	cv2.rectangle(draw_img,box[0],box[1],(0,0,255),6) 
                    boxes.append(box)
        heat = np.zeros_like(copied_image[:,:,0]).astype(np.float)
        heat = self.smoothening.add_heat(heat,boxes)
        # Apply threshold to help remove false positives
        heat = self.smoothening.apply_threshold(heat,1)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        copied_image = self.smoothening.draw_labeled_bboxes(np.copy(copied_image), labels)
        if (withboxes == True):           	
            return copied_image, boxes
        else:

            return copied_image