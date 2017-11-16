# Udacity Path Planning Project
Self-Driving Car Engineer Nanodegree Program
   
[//]: # (Image References)
[image1]: ./output_images/30-meters.png
[image2]: ./output_images/Completion-without-incident.png
[image3]: ./output_images/left_lane_change.png
[image4]: ./output_images/left_lane_change_2.png
[image5]: ./output_images/No_lane_change_due_to_car_back_car.png
[image6]: ./output_images/Reduced_velocity-due-to-front-car.png
[image7]: ./output_images/right_lane_change.png
[image8]: ./output_images/right_lane_change_without_incidents.png



The goals for the project are 

* To safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. 
* The car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. 
* The car drives < 50 MPH speed limit. The car avoids hitting other cars at all cost as well as driving inside of the marked road lanes at all times. 
* The car was able to drive wihtout incidents upto 15-20 miles. 
* Also the car did not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

The car lanes are marked as 
* Lane 0 - Leftmost Lane
* Lane 1 - Middle Lane
* Lane 2 - RightMost Lane 

* The car starts at Lane 1 with 0 velocity and gradulally increases the velocity at 2.24 mts/sec. 
* The maximum velocity of the car is set to 49.5 which is less than speed Limit of the highway.
* The car uses either the two points either from current or previous path points along with 30, 60, 90 waypoints in Frenet Co-ordinates. 
* All these points are taken in vehicle co-ordinates.
* These are the 5 achor points and added to spline, which is a single header file with high performance. 
* These points acts as reference points. Now, with any "x" point, spline can provide the "y" point. So, for ex: To determine the targety at targetx=30, targety=s(targetx)
* The future path points is calcluated from the previous path points.
* The path points is calulated for 30 mts which are equally spaced across.
* The spacing "N" is calculated using 
  * N = target_distance*((0.02*49.5)/2.24) 
  * Where - Target_distance is calculated using the Pythagoras theorem.
* With spline, for each value of x, corresponding y value is calculated, and converted into vehicle reference points.
* This provides a smooth slope as per the image below

![alt text][image1]

### Rubric Point
## The car is able to drive at least 4.32 miles without incident.
* The sensor fusion data is being used for making sure, the car senses upto 30 mts in the front and 15 mts behind for any cars before making any lanes changes. This is depicted in the main.cpp at line numbers 274 and 288. 
* If there are any cars within this range, the flag is set and this avoids any lane changes.
* With this the car is able to make a necessary lane changes and able to complete the drive without any incident and maintain the speed below 49.5 MPH.

![alt text][image2]

## The car drives according to the speed limit.

The maximum velocity of the car is set to 49.5. If the car velocity is reached, the velocity is automatically reduced at 2.24 mts/sec. This is done in main.cpp at line number 327.

## Max Acceleration and Jerk are not Exceeded.
The car starts at 0 velocity and accelerated at 2.24 m/s until it reaches 49.5 MPH(Line 202 and line 329). This reduces the jerk and max accelerations.

## Car does not have collisions.
With proper lane checks while switching avoids the car with any collision. The car was able to drive 13 miles without collisions with proper lane changes.

![alt text][image8]

## The car stays in its lane, except for the time between changing lanes. 
All through the highway travel, car was within the lanes. If there was no cars, the car was moved into middle lane. (Main.cpp - Line 334 - 343)

## The car is able to change lanes
Car was able to change lanes without issues on various occasions.

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

## Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.

## Tips

A really helpful resource for doing this project and creating smooth trajectories was using http://kluge.in-chemnitz.de/opensource/spline/, the spline function is in a single hearder file is really easy to use.

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!


## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

