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
* I have used starter code provided from the Udacity for completion of the project. 

## Implementation
### Rubric Point - The Model : 

**Student describes their model in detail. This includes the state, actuators and update equations.**

* In this project, I have implemented Model Predictive Control to drive the car around the track. 
* Kinematic Vehicle Model with Actuator constraints are the foundation of model predictive controls. 
* This defines the future trajectory of the vehicle. 
* In order to develop an optimal controller, we need define the cost function that reduces the errors (Cross track error and Orientation error)

The vehicle model consists of:
#### State 

* x: Vehicle x position
* y: Vehicle y position
* ψ (psi): vehicle’s angle in radians from the x-direction (radians)
* ν: vehicle’s velocity
* cte: cross track error - Error between the center of the road and the vehicle's position as the cross track error
* eψ : orientation error 

#### Control/Accuator Inputs 

* δ: Steering Angle 
* a: Accelerator/Brakes

Below are state, actuators and update equations.

![alt text][image2]

Kinematic Model is used for its simplicity. 

* Setup everything for Model Predictive Controller
	* Define the duration of trajectory T by choosing N and dt.
	* Define the vehicle Model
	* Define Constraints with Actuator Limitations
	* Find Cost Function

Below Diagrams depict the algorithm.

![alt text][image3]

![alt text][image4]

![alt text][image5]


### Rubric Point - Timestep Length and Elapsed Duration (N & dt) : 

**Student discusses the reasoning behind the chosen N (timestep length) and dt (elapsed duration between timesteps) values. Additionally the student details the previous values tried.**

* Prediction horizon - T = N * dt
	* T = Duration over which future predictions are made
	* N = N is the number of timesteps in the horizon
	* dt = dt is how much time elapses between actuations

* Number of Timesteps -  N
	* Model Predictive Control is to optimize the control inputs: [δ,a]. An optimizer will tune these inputs until a low cost vector of control inputs is found. The length of this vector is determined by N.

A good approach to setting N, dt, and T is to first determine a reasonable range for T and then tune dt and N appropriately, keeping the effect of each in mind. 

The value of N and dt are below.

* N = 7
* dt = 0.1 // I tested with 0.3, 0.8,

T = 7 * 0.1 = .7Secs

The future predictiosn are done over 0.7 secs.

### Rubric Point

** Student discusses the reasoning behind the chosen N (timestep length) and dt (elapsed duration between timesteps) values. Additionally the student details the previous values tried ** 


Other Values tried

* N = 7
* dt = 0.3

[ Model Predictive Controller ] https://youtu.be/0KyGlH6_-Gk

* N = 7
* dt = 0.8

[ Model Predictive Controller ] https://youtu.be/1NwPzLOY6Gs





### Rubric Point - Polynomial Fitting and MPC Preprocessing : 

**A polynomial is fitted to waypoints. If the student preprocesses waypoints, the vehicle state, and/or actuators prior to the MPC procedure it is described.**

* The JSON data consists of the map co-ordinates (waypoints). However, we need Vehicle co-ordinates for MPC algorithm. The conversion was done in the main.cpp (Lines 124 - 130)
* Following formulas were used for conversion.
	* dx = X-Coordinate of the map - x-Coordinate of the vehicle;
    * dy = Y-Coordinate of the map - y-Coordinate of the vehicle;
    * x_vehicle_coordinates = dx * cos(-psi) - dy * sin(-psi);
    * y_vehicle_coordinates = dy * cos(-psi) + dx * sin(-psi);
* Third Order polynomial is calculated to the given x and y coordinates representing vehicle-coordinates. Main.cpp (Line 133)


### Rubric Point - Model Predictive Control with Latency : 

**The student implements Model Predictive Control that handles a 100 millisecond latency. Student provides details on how they deal with latency.**

* In a real car, an actuation command won't execute instantly - there will be a delay as the command propagates through the system.
* A realistic delay might be on the order of 100 milliseconds. A latency of 100ms was added to increase stability.
* The delay was added to the Kinematic Model and passed to MPC routine. The code for this is found in main.cpp (Lines 137 - 143)


### Rubric Point - Model Predictive Control with Latency : 

** A polynomial is fitted to waypoints.

If the student preprocesses waypoints, the vehicle state, and/or actuators prior to the MPC procedure it is described. **




## Simulation
### Rubric Point - The vehicle must successfully drive a lap around the track. : 

**No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).
The car can't go over the curb, but, driving on the lines before the curb is ok.**


Final Video: N = 7 and dt = 0.1 

[Model Predictive Control - 50 mph] - https://youtu.be/Se6wj49RDBE

[Model Predictive Control - 75 mph] - https://youtu.be/kLQQ3xiI3oI






