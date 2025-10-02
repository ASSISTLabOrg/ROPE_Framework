#ifndef ROPE_SWDRIVERS_H   
#define ROPE_SWDRIVERS_H

#include <iostream>
#include <fstream>
#include <string>
#include <numeric> 
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

namespace swDrivers{

typedef boost::math::interpolators::cardinal_cubic_b_spline<double> bspline;
typedef std::vector<Driver> driver_vector;

class swDriver{

private:

    std::string label;
    const bspline driver_function; // data, cast as a function!
    bspline build_spline(std::vector<double> values, double t0, double dt);

public: 

    swDriver(std::vector<double> values, double t0, double dt, std::string label);
    double operator() (double t);
};

};
#endif