# Extended Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)
[image1]: ./output_images/Kalman_Filter_Output.png
[image2]: ./output_images/Kalman_Filter_Output_Lidar_Only.png
[image3]: ./output_images/Kalman_Filter_Output_Radar_Only.png
[image4]: ./output_images/Kalman-Filter-Algorithm.png

The goals / steps of this project are the following:

* Utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements
* The moving object is a bicycle that travels around the vehicle.
* The simulated data is already provided.
* Kalman Filter will be used to track the bicycle's position and velocity.
* Calculate Positions (px, py) and Velocity(vx, vy) of the Bicycle using Kalman Filter.
* Calulate RMSE from the ground truth.
* Lesser the RMSE value, the accurate are the results. 
* Kalman Filter contains two parts:
  * Prediction
    * State can be calulated by following formulas.
      * x1 = F * x + u 
      * P1 = F * P * FT + Q

  * Measurement Update:
      * y = z − Hx0
      * S = H*P0*HT + R
      * K = P0*HT*Sinv
      * x = x0 + Ky
      * P = (I − KH)P0

      * x -  Previous Estimate
      * x1 - Current Estimate
      * F - State Transistion Matrix
      * U - Motion Vector (Zero in the current Scneraio becuase we will not know the internal state of Bicycle)
      * z - Measurement
      * H - Measurement Function
      * R - Measurement Noise ()
      * I - Identity Matrix
      * P - Uncertinity Covariance
      * HT - H Transpose

* The Daiagram below depicts the Kalman Filter alogrithm.

![alt text][image4]


I have used starter code provided from the Udacity for completion of the project.

1. Setup the project.
2. Update the tools.cpp to Calculate RMSE and Jacobian Matrix
3. Update the fusion.h and fusion.cpp to add the necessary functions and initializations.
4. Calculate RMSE, Velocity and Poistion using Radar only.
![alt text][image3]
5. Calculate RMSE, Velocity and Poistion using Lidar only.
![alt text][image2]
6. Calculate RMSE, Velocity and Poistion using both sensors (Radar and Lidar).
![alt text][image1]


