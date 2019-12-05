/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
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
	//	 x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	particles.clear();
	default_random_engine gen;  // RNG

	num_particles = 100;  // tweak as needed

	// Initialize particle vector
	for (int i=0; i < num_particles; i++) {
		// Sample x, y, theta from normal distribution
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);

		// Initialize a new particle
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);

		// Add new particle to particle vector
		particles.push_back(p);
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//	http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//	http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;  // RNG

	// Run prediction for all particles
	for (int i=0; i < num_particles; i++) {
		Particle& p = particles[i];

		// Add Gaussian noise to x, y, theta
		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);
		double x = dist_x(gen);
		double y = dist_y(gen);
		double theta = dist_theta(gen);

		// Motion model (bicycle model with 0 wheelbase?)
		double theta_f;
		double x_f;
		double y_f;
		if (yaw_rate < 1e-6) {  // some small number close to 0, effectively yaw_rate==0
			theta_f = theta;
			x_f = x + velocity * delta_t * cos(theta);
			y_f = y + velocity * delta_t * sin(theta);
		} else {  // yaw_rate != 0
			theta_f = theta + yaw_rate * delta_t;
			x_f = x + (velocity/yaw_rate) * (sin(theta_f) - sin(theta));
			y_f = y + (velocity/yaw_rate) * (cos(theta) - cos(theta_f));
		}

		// Update values in particle
		p.x = x_f;
		p.y = y_f;
		p.theta = theta_f;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//	 observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//	 implement this method and use it as a helper during the updateWeights phase.

	// First, mark all observation ID's to -1, so we know which observations have associated landmark
	for (auto& obs : observations) {
		obs.id = -1;
	}

	// For each landmark, do nearest neighbor search to find closest observation,
	// and assign that observation's landmark ID accordingly
	for (int i=0; i < predicted.size(); i++) {
		double nearest_dist = -1.0;
		int nearest_obs = -1;
		LandmarkObs& lm = predicted[i];

		for (int j=0; j < observations.size(); j++) {
			LandmarkObs& obs = observations[j];

			// Distance between landmark and observation
			double distance = dist(lm.x, lm.y, obs.x, obs.y);

			// Keep track of nearest neighbor
			if (distance < nearest_dist || nearest_dist == -1) {
				nearest_dist = distance;
				nearest_obs = j;
			}
		}

		// Associate current landmark to one observation
		if (nearest_obs != -1) {
			observations[nearest_obs].id = lm.id;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//	 more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//	 according to the MAP'S coordinate system. You will need to transform between the two systems.
	//	 Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//	 The following is a good resource for the theory:
	//	 https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//	 and the following is a good resource for the actual equation to implement (look at equation
	//	 3.33
	//	 http://planning.cs.uiuc.edu/node99.html
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	// For each particle (i.e. hypothesis on where the ego vehicle is)
	for (int i=0; i < particles.size(); i++) {
		Particle& p = particles[i];

		// For each observation, convert observed landmark coordinates into map coordinates,
		// assuming that the ego vehicle is located where the particle is
		vector<LandmarkObs> observations_mc;  // vector of observations in map coordinates
		for (int j=0; j < observations.size(); j++) {
			LandmarkObs& obs = observations[j];
			LandmarkObs obs_mc;

			obs_mc.id = obs.id;
			obs_mc.x = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y;
			obs_mc.y = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y;

			observations_mc.push_back(obs_mc);
		}

		// Convert map_landmarks.landmark_list to vector<LandmarkObs>
		vector<LandmarkObs> predicted;
		for (auto& map_lm : map_landmarks.landmark_list) {
			LandmarkObs lm;

			lm.id = map_lm.id_i;
			lm.x = map_lm.x_f;
			lm.y = map_lm.y_f;

			predicted.push_back(lm);
		}

		// Associate observations to map landmarks (everything is in map coordinates)
		dataAssociation(predicted, observations_mc);

		// Calculate weight of this particle
		bool has_obs = false;
		double weight = 1.0;

		// For each observation with an associated landmark,
		// calculate that observation's weight contribution,
		// and multiply to particle's final weight
		for (auto& obs : observations_mc) {
			if (obs.id != -1) {
				// Get coordinates of observation
				double& x_obs = obs.x;
				double& y_obs = obs.y;

				// Find observation's associated landmark, extract its coordinates
				Map::single_landmark_s& lm = map_landmarks.landmark_list[0];  // FIXME
				for (auto& map_lm : map_landmarks.landmark_list) {
					if (map_lm.id_i == obs.id) {
						lm = map_lm;
					}
				}
				double mu_x = (double) lm.x_f;
				double mu_y = (double) lm.y_f;

				double exponent = (pow(x_obs - mu_x, 2))/(2*pow(sig_x, 2)) + pow(y_obs - mu_y, 2)/(2*pow(sig_y, 2));
				double obs_weight = exp(-exponent) / (2 * M_PI * sig_x * sig_y);

				const double OBS_WEIGHT_THRESH = 1e-4;  // prevent weights from going completely to 0
				if (obs_weight < OBS_WEIGHT_THRESH) {
					obs_weight = OBS_WEIGHT_THRESH;
				}

				weight *= obs_weight;

				has_obs = true;
			}
		}

		if (has_obs) {
			p.weight = weight;
		} else {
			cout << "WARNING: This particle has no valid observations!\n";
			p.weight = 0.0;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//	 http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> new_particles; // new particle array
	vector<double> weights;  // vector of weights, indexed by particle index

	for (int i=0; i < particles.size(); i++) {
		weights.push_back(particles[i].weight);
	}

	default_random_engine generator;  // RNG
	discrete_distribution<int> distribution(weights.begin(), weights.end());  // discrete distribution based on particle weights

	// Sample from discrete distribution
	for (int i=0; i < particles.size(); i++) {
		int random_idx = distribution(generator);

		new_particles.push_back(particles[random_idx]);
	}

	// Replace original particle vector
	particles = new_particles;  // TODO: check if there is memory leak?
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);	// get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);	// get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);	// get rid of the trailing space
	return s;
}
