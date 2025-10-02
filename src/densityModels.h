#ifndef ROPE_DENSITYMODELS_H   
#define ROPE_DENSITYMODELS_H

#include <iostream>
#include <fstream>
#include <string>
#include <numeric> 
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint.hpp>
#include "sindyFeatures.h"

namespace densityModels{

using sindyFeatures::Library;
using boost::numeric::ublas::prod;
using boost::numeric::odeint::integrate;

typedef boost::numeric::ublas::vector<double> state_vector;
typedef boost::numeric::ublas::matrix<double> matrix;

void write_solution( const state_vector &x , const double t );


//====================================== Generic model ===============================================

class DensityModel {
public:
    virtual void predict();
    virtual void get_name();
};

//====================================== SINDYc model ===============================================

class SINDYc : public DensityModel
{

    // Models a system of ordinary differential equations X'_(n,1) = A_(n,m) * F(X,U)_(m,1)
    // Where X is the system state and U are the control inputs

private:

    // Class attributes
    double n_variables = 0; // Length of state vector x = (x_1, x_2, ... , x_n), where x_i are the n independent variables of the system
    double n_controls = 0; // Length of control vector u = (u_1, u_2, ... , u_s) where u_i are the s external control variables
    double n_features = 0; // Length of library of features F = (f_1(X,U), f_2, ... , f_m), where f_i are the m features used to model the system
    Library library = Library(); // library of features F = (f_1(X,U), f_2, ... , f_m), where f_i are the m features used to model the system
    driver_vector U = {}; // contains the drivers...
    matrix A; // A = matrix(n,m), containing the coefficients

    // Push method for prediction
    void push(const state_vector &x , state_vector &dxdt , double /* t */);

    // Coefficient matrix builder
    matrix build_coeffs(std::vector<std::vector<double>> coeffs);

public:

    // Constructors
    SINDYc(std::string config_file);
    SINDYc(double n, double s, std::vector<std::string> features, std::vector<std::vector<double>> coeffs, driver_vector drivers);

    // Prediction method
    void predict(vec& initial_state, double t_start, double t_end, double dt);

    // Inspection methods
    void list_features();

};

// class DMDc : public DensityModel
// {
// private:
// public:
//     // Constructors
//     DMDc(std::string config_file);

//     // Predictor
//     void predict(vec& initial_state, double t_start, double t_end, double dt);
// };

// class LSTM : public DensityModel 
// {
// private:
// public:
//     // Constructors
//     LSTM(std::string config_file);

//     // Prediction method
//     void predict(vec& initial_state, double t_start, double t_end, double dt);
// };

};

#endif
