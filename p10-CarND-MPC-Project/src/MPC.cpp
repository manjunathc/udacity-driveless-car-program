// Algorithm for Solving MPC
//Begin State Feedback Loop
//    1. First Pass the current state to Model Predictive Controller
//    2. Next the Optimization Solver is called
//    3. Solver uses the initial step, the model, constraints and cost functions to return the vector of control inputs that minimize the cost function.
//    4. The solver will use IPOPT.
//    5. Apply the first control input to the vehicle and repeat .

#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
//size_t N = 20;  //N is the number of timesteps in the horizon. 
size_t N = 7;  //N is the number of timesteps in the horizon. This was modifed as per the review comments
double dt = 0.1;  //dt is the time elapses between actuations - Tested with 0.3, 0.8

const double reference_cte = 0.0;
const double reference_epsi = 0.0;
const double reference_v = 60.0;  // Set the Vehicle speed to 70 MPH, so the vehicle doesn't stop in between.

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.

// LF is a variable which measures the distance between the front of the vehicle and its center of gravity.
// The larger the vehicle , the slower the turn rate.
const double Lf = 2.67;

//Define the vehicle Model. [x,y,ψ(psi),v,cte,eψ(epsi)]. [δ(delta),a].

//The fg stores cost value.
//vars stores all variables used by the cost function and model.
//All these values are stored in a single vecotor Vars. This vector contains all variables used by the cost function and model: start at 0 to N.
//x_start starts with 0, y_start = x_start+N, psi_start = y_start + N.. so on.

//Below are the state Inputs
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;

//Below are the Error Inputs
size_t cte_start = v_start + N;  //Cross Track Error
size_t epsi_start = cte_start + N;  //Orientation Error

// Define Vehicle Contraints
size_t delta_start = epsi_start + N;  // Steering angle
size_t a_start = delta_start + N - 1;  //acceleration

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) {
    this->coeffs = coeffs;
  }

  //Cost Function

  typedef CPPAD_TESTVECTOR(AD<double>)ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    ////Ipopt expects fg to store the cost value at index 0, we sum all the components of the cost and store them at index 0.
    // The part of the cost based on the reference state.
    fg[0] = 0;
    for (int t = 0; t < N; t++) {
      fg[0] += CppAD::pow(vars[cte_start + t] - reference_cte , 2);
      fg[0] += 10 * CppAD::pow(vars[epsi_start + t] - reference_epsi, 2);  // Added a
      //fg[0] += CppAD::pow(vars[epsi_start + t] - reference_epsi, 2);
      fg[0] += CppAD::pow(vars[v_start + t] - reference_v, 2);
    }

    // Minimize change-rate.
    for (int t = 0; t < N - 1; t++) {
      fg[0] += CppAD::pow(vars[delta_start + t], 2);
      fg[0] += 50 * CppAD::pow(vars[a_start + t], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (int t = 0; t < N - 2; t++) {
      fg[0] += 1250 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += 5 * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    //Initialization & constraints
    //We initialize the model to the initial state. Recall fg[0] is reserved for the cost value, so the other indices are bumped up by 1.

    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    //State Vector
    //Xt+1   = xt + vt*cos(psit)*dt
    //yt+1   = yt +  vt*sin(psit)*dt
    //PSIt+1 = PSIt+vt/lf*delta*dt
    //Vt+1 = vt + at*dt

    //CTEt+1 = CTEt + Vt * sin(psi) *dt -  //Cross Track Error (CTE):
    //EPSIt+1 = EPSIt+vt/lf *delta * dt Orientation Error

    for (int t = 0; t < N - 1; t++) {
      // The state at time t+1 .
      AD<double> x1 = vars[x_start + t + 1];
      AD<double> y1 = vars[y_start + t + 1];
      AD<double> psi1 = vars[psi_start + t + 1];
      AD<double> v1 = vars[v_start + t + 1];
      AD<double> cte1 = vars[cte_start + t + 1];
      AD<double> epsi1 = vars[epsi_start + t + 1];

      // The state at time t.
      AD<double> x0 = vars[x_start + t];
      AD<double> y0 = vars[y_start + t];
      AD<double> psi0 = vars[psi_start + t];
      AD<double> v0 = vars[v_start + t];
      AD<double> cte0 = vars[cte_start + t];
      AD<double> epsi0 = vars[epsi_start + t];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + t];
      AD<double> a0 = vars[a_start + t];

      AD<double> f0 = 0.0;
      for (int i = 0; i < coeffs.size(); i++) {
        f0 += coeffs[i] * CppAD::pow(x0, i);
      }

      AD<double> psides0 = 0.0;
      for (int i = 1; i < coeffs.size(); i++) {
        psides0 += i*coeffs[i] * CppAD::pow(x0, i-1);  // f'(x0)
      }
      psides0 = CppAD::atan(psides0);

      // Here's `x` to get you started.
      // The idea here is to constraint this value to be 0.
      //
      // Recall the equations for the model:
      // x_[t] = x[t-1] + v[t-1] * cos(psi[t-1]) * dt
      // y_[t] = y[t-1] + v[t-1] * sin(psi[t-1]) * dt
      // psi_[t] = psi[t-1] + v[t-1] / Lf * delta[t-1] * dt
      // v_[t] = v[t-1] + a[t-1] * dt
      // cte[t] = f(x[t-1]) - y[t-1] + v[t-1] * sin(epsi[t-1]) * dt
      // epsi[t] = psi[t] - psides[t-1] + v[t-1] * delta[t-1] / Lf * dt
      fg[2 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[2 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[2 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[2 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[2 + cte_start + t] = cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));

      fg[2 + epsi_start + t] = epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);

    }

  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {
}
MPC::~MPC() {
}

/*
 *
 * 1. Define the duration of trajectory T by choosing N and dt
 2. Define the vehicle Model
 3. Define Constraints with Actuator Limitations
 4. Find Cost Function
 *
 */

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {

  std::cout << "MPC::Solve 1 -->" << std::endl;

  bool ok = true;
  typedef CPPAD_TESTVECTOR(double)Dvector;

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9

  // Based on the above formula, n_vars can be calculated as below.
  // 4 * N + 2 * N-1
  // [x,y,ψ(psi),v]. [δ(delta),a].
  size_t n_vars = N * 6 + (N - 1) * 2;

  // TODO: Set the number of constraints
  // [x,y,ψ(psi),v,cte,eψ(epsi)].
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }
  std::cout << "MPC::Solve 2 -->" << std::endl;

  //Define the vehicle Model

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.

  //Initialize [x,y,ψ(psi),v,cte,eψ(epsi)] with Lower and Upper Bounds
  for (int i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e5;
    vars_upperbound[i] = 1.0e5;
  }

  //Define Constraints with Actuator Limitations
  //Set the constraints delta (Steering angle to -25 degrees to +25 Degrees)
  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  //Set the constraints delta (Brake/acclerator angle to -1 degrees to +1 Degrees)
  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  std::cout << "MPC::Solve 3 -->" << std::endl;
  constraints_lowerbound[x_start] = state[0];
  constraints_lowerbound[y_start] = state[1];
  constraints_lowerbound[psi_start] = state[2];
  constraints_lowerbound[v_start] = state[3];
  constraints_lowerbound[cte_start] = state[4];
  constraints_lowerbound[epsi_start] = state[5];

  //Pass the current state to Model Predictive Controller

  constraints_upperbound[x_start] = state[0];
  constraints_upperbound[y_start] = state[1];
  constraints_upperbound[psi_start] = state[2];
  constraints_upperbound[v_start] = state[3];
  constraints_upperbound[cte_start] = state[4];
  constraints_upperbound[epsi_start] = state[5];
  std::cout << "MPC::Solve 4 -->" << std::endl;

  FG_eval fg_eval(coeffs);
  std::cout << "MPC::Solve 5 -->" << std::endl;

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(options, vars, vars_lowerbound,
                                        vars_upperbound, constraints_lowerbound,
                                        constraints_upperbound, fg_eval,
                                        solution);

  std::cout << "MPC::Solve 6 -->" << std::endl;

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  //Store the values in the vector

  std::cout << "MPC::Solve 7 -->" << std::endl;

  mpc_x = {};
  mpc_y = {};
  for (int i = 0; i < N; i++) {
    mpc_x.push_back(solution.x[x_start + i]);
    mpc_y.push_back(solution.x[y_start + i]);
  }

  vector<double> final_result;
  final_result.push_back(solution.x[delta_start]);
  final_result.push_back(solution.x[a_start]);

  return final_result;
}
