#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() {
  return M_PI;
}
double deg2rad(double x) {
  return x * pi() / 180;
}
double rad2deg(double x) {
  return x * 180 / pi();
}

const double dt = 0.1;  // time step duration dt in s
const double LF = 2.67; //

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
      uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
      string sdata = string(data).substr(0, length);
      cout << sdata << endl;
      if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
        string s = hasData(sdata);
        if (s != "") {
          auto j = json::parse(s);
          string event = j[0].get<string>();
          if (event == "telemetry") {
            // j[1] is the data JSON object

            //Capture the State
            vector<double> ptsx = j[1]["ptsx"];// The global x positions of the Map-coordinates.
            vector<double> ptsy = j[1]["ptsy"];// The global y positions of the Map-coordinates.
            double px = j[1]["x"];// The global x position of the vehicle.
            double py = j[1]["y"];//  The global y position of the vehicle.
            double psi = j[1]["psi"];// The orientation of the vehicle
            double v = j[1]["speed"];// Speed of the vehicle
            v = v * 0.447;// mph to m/s

            //Capture the actuator Inputs
            //Obtain steering angle and throttle from JSON Data
            double steering_angle = j[1]["steering_angle"];
            double throttle = j[1]["throttle"];

            //Derive a Kinematic Model over time dt using below equations

            //xt+1= x + v*cos(psi)*dt and
            //yt+1 = y + v*sin(psi)*dt
            //psit+1 =psi+v/lf * delta * dt
            //vt+1 = vt + at*dt

            //dt is the difference between time t+1 and time t. The value is chosen to 0.1
            //LF is a variable which measures the distance between the front of the vehicle and its center of gravity. The value is chosen to be 2.67

            //First convert all the ptsx,ptsy positions(Map Co-ordinates) to vehicle co-ordinates. The below code will convert all Map co-ordinates to vehicle co-ordinates.
            Eigen::VectorXd x_vehicle_coordinates(ptsx.size());
            Eigen::VectorXd y_vehicle_coordinates(ptsx.size());
            for(int i = 0; i < ptsx.size(); i++) {
              const double dx = ptsx[i] - px;
              const double dy = ptsy[i] - py;
              x_vehicle_coordinates[i] = dx * cos(-psi) - dy * sin(-psi);
              y_vehicle_coordinates[i] = dy * cos(-psi) + dx * sin(-psi);
            }

            // Fit a 3rd order polynomial to the given x and y coordinates representing map-coordinates.
            auto coeffs = polyfit(x_vehicle_coordinates, y_vehicle_coordinates, 3);
            const double cte = coeffs[0];
            const double epsi = -atan(coeffs[1]);//-f'(0)

            // Kinematic model is used to predict vehicle state at the actual moment of control (current time + delay dt)
            const double px_act = v * dt;
            const double py_act = 0;
            const double psi_act = - v * steering_angle * dt / LF;
            const double v_act = v + throttle * dt;
            const double cte_act = cte + v * sin(epsi) * dt;
            const double epsi_act = epsi + psi_act;
            Eigen::VectorXd state(6);
            state << px_act, py_act, psi_act, v_act, cte_act, epsi_act;

            /*
             * TODO: Calculate steering angle and throttle using MPC.
             *
             * Both are in between [-1, 1].
             *
             */

            double steer_value;
            double throttle_value;

            /* Begin State Feedback Loop
                1. First Pass the current state to Model Predictive Controller
                2. Next the Optimization Solver is called
                3. Solver uses the initial step, the model, constraints and cost functions to return the vector of control inputs that minimize the cost function.
                4. The solver will use IPOPT.
                5. Apply the first control input to the vehicle and repeat the loop.*/

            vector<double> mpc_results = mpc.Solve(state, coeffs);
            steer_value = mpc_results[0]/ deg2rad(25);// converts the degrees to Radians.
            throttle_value = mpc_results[1];

            json msgJson;
            // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
            // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].

            //The yellow line is the reference trajectory and the green line the trajectory computed by Model Predictive Control.
            //In this example the horizon has 20 steps, N, and the space in between white pebbles signifies the time elapsed, dt which is 0.1.

            msgJson["steering_angle"] = -steer_value;
            msgJson["throttle"] = throttle_value;

            //Display the MPC predicted trajectory
            vector<double> mpc_x_vals = mpc.mpc_x;
            vector<double> mpc_y_vals = mpc.mpc_y;

            //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
            // the points in the simulator are connected by a Green line

            msgJson["mpc_x"] = mpc_x_vals;
            msgJson["mpc_y"] = mpc_y_vals;

            //Display the waypoints/reference line
            vector<double> next_x_vals;
            vector<double> next_y_vals;

            //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
            // the points in the simulator are connected by a Yellow line
            for(int i = 0; i<ptsx.size();i++) {
              next_x_vals.push_back(x_vehicle_coordinates[i]);
              next_y_vals.push_back(y_vehicle_coordinates[i]);
            }

            msgJson["next_x"] = next_x_vals;
            msgJson["next_y"] = next_y_vals;

            auto msg = "42[\"steer\"," + msgJson.dump() + "]";
            std::cout << msg << std::endl;
            // Latency
            // The purpose is to mimic real driving conditions where
            // the car does actuate the commands instantly.
            //
            // Feel free to play around with this value but should be to drive
            // around the track with 100ms latency.
            //
            // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
            // SUBMITTING.
            this_thread::sleep_for(chrono::milliseconds(100));
            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          }
        } else {
          // Manual driving
          std::string msg = "42[\"manual\",{}]";
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      }
    });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
      size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
      char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
h.run();
}
