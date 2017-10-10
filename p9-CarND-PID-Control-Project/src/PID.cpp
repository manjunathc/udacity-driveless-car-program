#include "PID.h"

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	this->Kp = Kp; // tau P
	this->Ki = Ki; //tau I
	this->Kd= Kd;  // tau d

    i_error = 0;
    d_error = 0;
    p_error = 0;
}

void PID::UpdateError(double cte) {

	d_error = cte - p_error; //set the differential error - (diff-cte)
	p_error = cte; //Set cross track error to process error - (cte)
	i_error += cte ; //set the Integral error - (int+sum of all cte)

}

double PID::TotalError() {
	return d_error+i_error+p_error;
}
