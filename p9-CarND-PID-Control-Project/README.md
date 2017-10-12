# Udacity PID Controller Project
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)
[image1]: ./output_images/compiled_code.png
[image2]: ./output_images/generic_cmake_list.png

* In this project, I have used PID controller to maneuver the vehicle around the track. The simulator will provide the cross track error (CTE) and the velocity (mph) in order to compute the appropriate steering angle. 


The goals / steps of this project are the following:

* Maneuver the vehicle around the track.
* I have used two PID contollers with one controlling the Steering Value and other for Speed.
* The hyper parameters have been set to both the contollers.
* Cross Track Error(CTE) and Speed error are calculated with the respective formula..
* With two PID controllers, I was able to Maneuver the vehicle without going off the track. 
* Twiddle was not needed for this project.


I have used starter code provided from the Udacity for completion of the project.

## Rubric Point : Your code should compile.

**Code must compile without errors with cmake and make.

Given that we've made CMakeLists.txt as general as possible, it's recommend that you do not change it unless you can guarantee that your changes will still compile on any platform.**

* Code is compiled as per the screen shot below and CMakeLists.txt is generic.

![alt text][image1]


![alt text][image2]



## Rubric Point : Describe the effect each of the P, I, D components had in your implementation.

**Student describes the effect of the P, I, D component of the PID algorithm in their implementation. Is it what you expected? 
Visual aids are encouraged, i.e. record of a small video of the car in the simulator and describe what each component is set to.**

**Videos are created for each values**

I have used two PID contollers with one controlling the Steering Value and other for Speed. I have set the maximum speed for 50mph for CTE = 0. I manually tuned the parameters for both PID controllers.

P = Process
I = Integral
D = Defivative

A proportional–integral–derivative controller (PID controller or three term controller) is a control loop feedback mechanism widely used in industrial control systems and a variety of other applications requiring continuously modulated control. [Wikipedia] https://en.wikipedia.org/wiki/PID_controller. The proportional, integral, and derivative terms are summed to calculate the output of the PID controller. 

**Cross Track error** is the lateral distance between the vehicle and reference Tracjectory (CTE).

#### Steering Value - alpha(Controll Output) = -tau_p * CTE - tau_d * diff_CTE - tau_i * int_CTE 

Here, 
**CTE** - Cross Track Error
**tau_p = Kp**
**tau_d = Kd**
**tau_i = Ki**

The proportion P is calcuated by 

**P = Process Term**

-tau_p*CTE - Where tau_p(Kp) the proportional Gain constant. 

A high proportional gain results in a large change in the output for a given change in the error. If the proportional gain is too high, the system can become unstable. For ex: in the current case if the Kp is changed from 0.1 to 0.5 the car oscillates faster and eventually overshoots and will be out of the track.

**D = Differential Term**

-tau_d* d/dt(CTE)
where 

tau_d = differential gain
d/dt(CTE) = CTE(t) - CTE(t-1)/delta_t (delta_t = 1)

To reduce Overshoot, it needs to counter steer and create a negative effect and gracefully approach target trajectory. 
In the above equaton, tau_d (Kd) is the differential gain and its inversely proportional to the proportional Gain thus reducing the effect of Proportional gain. 

**I = Integral term**

-tau_i(ki) * Integral (Sum of all Cross Track Error)

The integral in a PID controller is the sum of the instantaneous error over time and gives the accumulated offset that should have been corrected previously.The accumulated error is then multiplied by the integral gain (Ki) and added to the controller output. [Wikipedia] https://en.wikipedia.org/wiki/PID_controller

Ki is the integral gain. The integral term accelerates the movement of the process towards setpoint and eliminates the residual steady-state error that occurs with a pure proportional controller. It mainly used to reduce the systematic Bias(Big Cross Track Error(CTE)).

## Rubric Point : Describe how the final hyperparameters were chosen: 
**Student discusses how they chose the final hyperparameters (P, I, D coefficients). This could be have been done through manual tuning, twiddle, SGD, or something else, or a combination!**

I used manual tuning approach as described in [Ziegler–Nichols] https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method

### First kept Ki and Kd to zero. Below is the video at Kp=0.3
[PID Controller with High Kp oscillations and Zero Ki,Kd] 

https://youtu.be/Xw4pIwU0PVc

### Next, I modifed the value of Ki = 0.0001 and Kd = 3.0. Modified Kp from 0.1 to 0.2 and 0.3. At 0.3 it started high oscillations.
Below is the video for high Kp of 0.3

[PID Controller with High Kp oscillations] 

https://youtu.be/PQmTFTotB78

### Below is the video for Low Kp of 0.0 = This causes a big CTE. Please see the below Video.
[PID Controller with Zero KD oscillations] 

https://youtu.be/6z3T3TSD_J4

### Below is the video for high Ki = 0.1 which causes Overshoots.
[PID Controller with Ki = 0.1] 

https://youtu.be/0aCd7cLq378

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


Rubric 

Simulation

## CRITERIA : The vehicle must successfully drive a lap around the track. No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).

Project Details: 
1. Setup the project.
2. Update the PID.cpp for calulations and main.cpp to invoke PID, to calulate the Cross Track Error, Speed Error, Steering Value and Throttle.
3. Run the simuator to capture the simulator Video.

### Final Video - with Succesful tuning of parameters.

[PID Controller] 

https://youtu.be/cN4xRQOgbTA



