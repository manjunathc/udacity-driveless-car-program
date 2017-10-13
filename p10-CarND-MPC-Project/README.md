# Udacity Model Predictive Control Project
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)
[image1]: ./output_images/MPC_Algorithm_PNG.png
[image2]: ./output_images/vehicle_model_equation.png
[image3]: ./output_images/MPC_Algorithm_Step1.png
[image4]: ./output_images/MPC_Algorithm_Step2.png
[image5]: ./output_images/MPC_Algorithm_Step3.png
[image6]: ./output_images/Vehicle_State.png 

The goals / steps of this project are the following:

* Maneuver the vehicle around the track.
* Calculate the Cost Funtion, Optmial Trajectory, Steering angle (δ) and acceleration/throttle (a) using - 
	* JSON data with Map-Coordinates, Vehicle Model which consists of State and Control Inputs.
	* State inputs are [x,y,ψ,v,cte,eψ] and Control Inputs(Actuators)


* I have used MPC to optimize the Control Inputs, simulate tracjectory and minimize the errors. (Cross Track Error and Orientation error).
* Kinematic Model is used for its simplicity. However, it doesn't take into account the external forces such as gravity, tire forces, Mass.



I have used starter code provided from the Udacity for completion of the project. 


## Rubric Point : Your code should compile.

**Code must compile without errors with cmake and make. Given that we've made CMakeLists.txt as general as possible, it's recommend that you do not change it unless you can guarantee that your changes will still compile on any platform.**

* Code is compiled as per the screen shot below and CMakeLists.txt is generic.


## Rubric Point - The Model : 

**Student describes their model in detail. This includes the state, actuators and update equations.**

* In this project, I have implemented Model Predictive Control to drive the car around the track. 
* Kinematic and Dynamic Vehicle Model with Actuator constraints are the foundation of model predictive controls. 
* This defines the future trajectory of the vehicle. 
* In order to develop an optimal controller, we need define the cost function that reduces the errors (Cross track error and Orientation error)

State diagram is as shown below.

![alt text][image3]

The vehicle model consists of:
#### State 

* x: Vehicle x position
* y: Vehicle y position
* ψ (psi): vehicle’s angle in radians from the x-direction (radians)
* ν: vehicle’s velocity
* cte: cross track error - Error between the center of the road and the vehicle's position as the cross track error
* eψ : orientation error 

#### Control/Accuator Inputs 

*δ: Steering Angle 
*a: Accelerator/Brakes

![alt text][image2]

Kinematic Model is used for its simplicity. The Model setup is depicted in the below diagram.

![alt text][image1]

* Setup everything for Model Predictive Controller
	* Define the duration of trajectory T by choosing N and dt.
	* Define the vehicle Model
	* Define Constraints with Actuator Limitations
	* Find Cost Function

Below Diagrams depict the algorithm.

![alt text][image3]

![alt text][image4]

![alt text][image5]


## Rubric Point - Timestep Length and Elapsed Duration (N & dt) : 

**Student discusses the reasoning behind the chosen N (timestep length) and dt (elapsed duration between timesteps) values. Additionally the student details the previous values tried.**

* prediction horizon - T = N * dt
	*T = Duration over which future predictions are made
	*N = N is the number of timesteps in the horizon
	*dt = dt is how much time elapses between actuations

* Number of Timesteps -  N
	*Model Predictive Control is to optimize the control inputs: [δ,a]. An optimizer will tune these inputs until a low cost vector of control inputs is found. The length of this vector is determined by N.

A good approach to setting N, dt, and T is to first determine a reasonable range for T and then tune dt and N appropriately, keeping the effect of each in mind. 



