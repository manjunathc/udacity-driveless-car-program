# Udacity PID Controller Project
Self-Driving Car Engineer Nanodegree Program

* In this project, I have used PID controller to maneuver the vehicle around the track. The simulator will provide the cross track error (CTE) and the velocity (mph) in order to compute the appropriate steering angle. 


The goals / steps of this project are the following:

* Maneuver the vehicle around the track.
* I have used two PID contollers with one controlling the Steering Value and other for Speed.
* The hyper parameters have been set to both the contollers.
* Cross Track Error(CTE) and Speed error are calculated with the respective formula..
* With two PID controllers, I was able to Maneuver the vehicle without going off the track. 
* Twiddle was not needed for this project.


I have used starter code provided from the Udacity for completion of the project.

Rubric Point : Describe the effect each of the P, I, D components had in your implementation.

1. I have used two PID contollers with one controlling the Steering Value and other for Speed. I have set the maximum speed for 50mph for CTE = 0. I manually tuned the parameters for both PID controllers.

Rubric Point : Describe how the final hyperparameters were chosen

1. The gains were found better with
  Kp = 0.1;
  Ki = 0.0001;
  Kd = 3.0;
2. For Speed PID the gains were found better with 
  sKp = 0.5;
  sKi = 0.0001;
  sKd = 4.0;
3. I modified Kd values to tune the Oscillation and Ki to reduce the steady-state error.
4. I tried to use twiddle, however it was not needed as the car was performing better. 
5. For simulator, I used input of (640x480) with simple Graphics quality and recorded using the phone and uploaded to youtube.


Project Details:
1. Setup the project.
2. Update the PID.cpp for calulations and main.cpp to invoke PID, to calulate the Cross Track Error, Speed Error, Steering Value and Throttle.
3. Run the simuator to capture the simulator Video.

Below is the video 

[PID Controller] https://youtu.be/cN4xRQOgbTA



