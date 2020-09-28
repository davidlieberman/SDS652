#include <vector>
#include <iostream>
#include <random>

#include "smc.h"

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> runif(0, 100);
double MEAN = 0, STDDEV = 1;
normal_distribution<double> rnorm(MEAN, STDDEV);

// https://stackoverflow.com/questions/7560114/random-number-c-in-some-range
Particle::Particle() {
    // Randomize position
    x = runif(gen);
    y = runif(gen);
}

void Particle::set_weight(double new_weight) {
    weight = new_weight;
}

void Particle::perturb() {
    x += rnorm(gen);
    y += rnorm(gen);
}

void Particle::update_pos(double new_x, double new_y) {
    x = new_x;
    y = new_y;
}

// N number of particles
SMC::SMC(int N) {
    _N = N;
    actual_x = actual_y = 0;
    for (int i = 0; i < N; i++) {
        Particle new_particle = Particle();
        new_particle.set_weight(1/N);
        particles.push_back(new_particle);
    }

    for (int i = 0; i < 10; i++) {
        landmarks.push_back(Particle());
    }
}


void SMC::predict(vector<double> action) {
    actual_x += action[0] + rnorm(gen);
    actual_y += action[1] + rnorm(gen);
    for (auto &particle : particles) {
        double x = action[0] + rnorm(gen);
        double y = action[1] + rnorm(gen);
        particle.update_pos(x, y);
    }
}

void SMC::update() {
    for (auto &landmark : landmarks) {
        for (auto &particle : particles) {
            double x_dist = pow(landmark.x - particle.x, 2);
            double y_dist = pow(landmark.y - particle.y, 2);
            double norm = sqrt(x_dist + y_dist);
            // TODO in the context of object tracking --> update according to actual distance to landmark + noise
            // Update importance weights according to perturbed observations
            // https://stats.stackexchange.com/questions/340066/how-to-calculate-importance-weights-for-update-step-of-an-sir-sequential-import
        }
    }
}


void SMC::resample() {
    vector<Particle> new_particles;
    for (int i = 0; i < _N; i++) {
        Particle new_particle = weighted_choice();
        new_particle.set_weight(1/_N);
        new_particles.push_back(new_particle);
    }
}

Particle SMC::weighted_choice() {
    uniform_real_distribution<double> unifsample(0, 1);
    double choice = unifsample(gen);
    double weight_sum = 0;
    for (auto &particle : particles) {
        weight_sum += particle.weight;
        if (choice <= weight_sum) {
            return particle;
        }
    }

    return particles[-1];

}


vector<double> SMC::average() {
    vector<double> sum{0, 0};
    double sum_weights = 0;
    for (auto &particle : particles) {
        sum[0] += particle.x * particle.weight;
        sum[0] += particle.y * particle.weight;
        sum_weights += particle.weight;
    }

    sum[0] /= sum_weights;
    sum[1] /= sum_weights;

    return sum;
}


void SMC::print(int N) {
    N = max(N, _N);
    for (int i = 0; i < N; i++) {
        particles[i].perturb();
        cout << particles[i].x << ", " << particles[i].y << '\n';
    }
}
