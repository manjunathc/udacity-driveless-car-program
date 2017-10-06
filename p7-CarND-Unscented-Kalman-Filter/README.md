# Unscented Kalman Filters Project
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)
[image1]: ./output_images/No_Tuning.png
[image2]: ./output_images/Radar_Only.png
[image3]: ./output_images/Lidar_Only.png
[image4]: ./output_images/Tuned.png
[image5]: ./output_images/UKF_Roadmap.png


* A standard Kalman filter can only handle linear equations. Both the extended Kalman filter and the unscented Kalman filter allow you to use non-linear equations; the difference between EKF and UKF is how they handle non-linear equations. 


The goals / steps of this project are the following:

* Utilize a unscented Kalman filter using the CTRV motion model to estimate the state of a moving object with complex turns and with noisy lidar and radar measurements
* The moving object is a bicycle that travels around the vehicle.
* The simulated data is already provided.
* Unscented Kalman filter Kalman Filter will be used to track the bicycle's position and velocity.
    * Initialization - The initialization process requires to tune the process noise and measurement noise, State Vector, Covariance Matrix, etc
    * Prediction - Process of Generating sigma Points, Predict Sigma Points and Predict Mean and Co-variance
    * Update - Predict Meaurement and Update the state.
  

* The Daiagram below depicts the Unscented Kalman Filter alogrithm.

![alt text][image5]


I have used starter code provided from the Udacity for completion of the project.

1. Setup the project.
2. Update the tools.cpp to Calculate RMSE
3. Update the UKF.h and UKF.cpp to add the necessary functions and initializations.
4. Update kalman_filter.cpp with ProcessMeasurement(), Prediction, UpdateRadar and UpdateLidar() funcitons.
5. Calculate RMSE, Velocity and Poistion without Tuning Parameters.
![alt text][image1]
5. Calculate RMSE, Velocity and Poistion using Radar only.
![alt text][image2]
6. Calculate RMSE, Velocity and Poistion using Lidar only.
![alt text][image3]
7. Calculate RMSE, Velocity and Poistion using both sensors (Radar and Lidar).
![alt text][image4]


