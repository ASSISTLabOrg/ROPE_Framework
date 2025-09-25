#ifndef SINDY_H   
#define SINDY_H

#include <iostream>
#include <fstream>
#include <string>
#include <numeric> 
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/odeint.hpp>
#include "features.h"

using features::Library;

typedef boost::numeric::ublas::vector<double> vec;
typedef boost::numeric::ublas::matrix<double> mat;

namespace sindy
{

class SINDYc
{
private:

    void push(const vec &x , vec &dxdt , double /* t */);

public:

    // Models a system of ordinary differential equations X'_(n,1) = A_(n,m) * F(X,U)_(m,1)
    // Where X is the system state and U are the control inputs

    // 
    double n_variables = 0; // Length of state vector x = (x_1, x_2, ... , x_n), where x_i are the n independent variables of the system
    double n_controls = 0; // Length of control vector u = (u_1, u_2, ... , u_s) where u_i are the s external control variables
    double n_features = 0; // Length of library of features F = (f_1(X,U), f_2, ... , f_m), where f_i are the m features used to model the system
    Library library = Library(); // library of features F = (f_1(X,U), f_2, ... , f_m), where f_i are the m features used to model the system
    mat A; // A = matrix(n,m), containing the coefficients

    //
    SINDYc(std::string config_file);
    SINDYc(double n, double s, std::vector<std::string> features, std::vector<std::vector<double>> coeffs);
    mat build_coeffs(std::vector<std::vector<double>> coeffs);
    void solve(vec& initial_state, double t_start, double t_end, double dt);

};

void write_solution( const vec &x , const double t );

};
#endif