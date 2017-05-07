import numpy as np

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = [np.array([False])]    
        #y values for detected line pixels
        self.ally = [np.array([False])]  
        self.initial_frame = True

        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        self.curvature = None

    def ln_updateLine(self,left_fitx,right_fitx):
        self.allx = left_fitx
        self.ally = right_fitx
        
    def ln_getLine(self):
        return self.allx,self.ally

    def ln_get_curvature(self,y_eval,left_fit,right_fit):
        left_curverad = ((1 + (2*left_fit[0]*y_eval*self.ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*self.ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        #print("Radius of Left Curvature: %f " % left_curverad)
        #print("Radius of Right Curvature: %f " % right_curverad)
        self.curvature = (left_curverad+right_curverad)/2
        return self.curvature

    def ln_get_offset_from_center(self,left_fit,right_fit):
        center = (((left_fit[0]*720**2+left_fit[1]*720+left_fit[2]) +(right_fit[0]*720**2+right_fit[1]*720+right_fit[2]) ) /2 - 640)*self.xm_per_pix
        return center

