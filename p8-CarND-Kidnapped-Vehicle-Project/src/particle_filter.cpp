/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// This line creates a normal (Gaussian) distribution for x.

	//random number engine class that generates pseudo-random numbers
	default_random_engine gen;

	//Initialize the total particles
	num_particles = 50;

	//GPS Provided Positions
	//creates a normal (Gaussian) distribution for Position x.
	normal_distribution<double> dist_x(x, std[0]);

	//creates a normal (Gaussian) distribution for Position y.
	normal_distribution<double> dist_y(y, std[1]);

	//creates a normal (Gaussian) distribution for Position theta.
	normal_distribution<double> dist_theta(theta, std[2]);

	//Initialize all particles
	//Each particle has id, x-coordinate, y-co-ordinate and angle theta and weight. Weight is set to 1 for all particles.

	for (int i = 0; i < num_particles; i++) {
		//Create a new particle
		Particle particle;

		//intialize the particle
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		//Save particles in particles vector
		particles.push_back(particle);
	}

	is_initialized = true;
	cout << "Init Ended " << endl;
	return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//Prediction step is used to update the velocity and Yaw Rate. Also add Gaussian Noise.
	//Following formulas are used in prediction step.

	//For yaw rate greater than zero:
	//xf = x_pos + (velocity/yaw_rate) * (sin(theta + yaw_rate*delta_t) - sin (theta))
	//yf = y_pos + (velocity/yaw_rate) * (cos (theta) - cos(theta + yaw_rate*delta_t))
	//theatf = theta + yaw_rate*delta_t

	//For yaw rate is equal or less than zero:
	//xf = x_pos + velocity * sin(theta) * delta_t;
	//yf = y_pos + velocity * cos(theta) * delta_t;
	//thetaf = theta;

	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {
		double xf = particles[i].x;
		double yf = particles[i].y;
		double theta = particles[i].theta;
		double thetaf = theta + yaw_rate * delta_t;

		if (fabs(yaw_rate) > 0.00001) {
			particles[i].x = xf
					+ velocity / yaw_rate * (sin(thetaf) - sin(theta));
			particles[i].y = yf
					+ velocity / yaw_rate * (cos(theta) - cos(thetaf));
			particles[i].theta = thetaf;
		} else {
			particles[i].x = xf + velocity * sin(theta) * delta_t;
			particles[i].y = yf + velocity * cos(theta) * delta_t;
			particles[i].theta = theta;
		}

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

	return;

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
		std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations,
		const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//Observations need to be converted to Map Co-ordinates.

	weights.clear();

	for (int i = 0; i < num_particles; i++) {

		//Particle co-ordinates
		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double theta_p = particles[i].theta;

		//Transform car observations to map coordinates supposing that the particle is the car.
		particles[i].associations.clear();
		particles[i].sense_x.clear();
		particles[i].sense_y.clear();
		double weight = 1;

		//Obtain observation for each Particle
		for (int j = 0; j < observations.size(); j++) {
			//Obtain Observation Co-ordinates
			double x_c = observations[j].x;
			double y_c = observations[j].y;

			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];

			//Compute Map Particle Co-ordinates
			// # transform to map x coordinate
			//Below computations will have observations have been transformed into the map's coordinate space
			double x_m = x_p + (x_c * cos(theta_p)) - (y_c * sin(theta_p));

			// # transform to map y coordinate
			double y_m = y_p + (x_c * sin(theta_p)) + (y_c * cos(theta_p));

			//The next step is to associate each transformed observation with a land mark identifier.
			//Calculate mu-x and mu-y - Co-ordinates to the nearby LandMark

			double range = 1000;
			int min_value = -1;
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				double l_x = map_landmarks.landmark_list[k].x_f;
				double l_y = map_landmarks.landmark_list[k].y_f;
				double mu_x = l_x - x_m;
				double mu_y = l_y - y_m;
				double delta_range = pow(pow(mu_x, 2) + pow(mu_y, 2), 0.5);
				if (delta_range < range) {
					//cout<<"Inside Delta" << endl;
					range = delta_range;
					min_value = k;
				}
			}

			double nearbyLandMark_x = map_landmarks.landmark_list[min_value].x_f; //mu_x
			double nearbyLandMark_y = map_landmarks.landmark_list[min_value].y_f; //mu_y

			//Calculate the weights of each particle using a mult-variate Gaussian distribution
			weight = weight* exp(- 0.5 * (pow((nearbyLandMark_x - x_m) / sig_x, 2) + pow((nearbyLandMark_y - y_m) / sig_y, 2))) / (2 * M_PI * sig_x * sig_y);

		}

		//Update the weights of each particle
		particles[i].weight = weight;
		weights.push_back(weight);
	}

	return;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	std::vector<Particle> resampled_particles;

	weights.clear();
	int k = -1;
	for (int i = 0; i < num_particles; i++) {
		k = distribution(gen);
		resampled_particles.push_back(particles[k]);
		weights.push_back(particles[k].weight);
	}

	particles = resampled_particles;

	return;

}

Particle ParticleFilter::SetAssociations(Particle particle,
		std::vector<int> associations, std::vector<double> sense_x,
		std::vector<double> sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
