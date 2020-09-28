#ifndef SDS652_SMC_H
#define SDS652_SMC_H

#include <vector>

using namespace std;

class Particle {
public:
    double x, y;
    double weight;
    Particle();
    void set_weight(double new_weight);
    void perturb();
    void update_pos(double new_x, double new_y);
};

class SMC {
public:
    vector<Particle> particles;
    int _N;

    double actual_x, actual_y;
    vector<Particle> landmarks;

    // N number of particles
    SMC(int N=1000);

    void predict(vector<double> action);
    void update();
    void resample();
    Particle weighted_choice();
    vector<double> average();
    void print(int N);
};


#endif //SDS652_SMC_H
