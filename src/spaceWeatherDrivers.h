#ifndef ROPE_SWDRIVERS_H   
#define ROPE_SWDRIVERS_H

#include <iostream>
#include <fstream>
#include <string>
#include <numeric> 
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

namespace SpaceWeatherDrivers{

class SpaceWeatherDriver{

private:

    std::string label;
    const bspline driver_function; // data, cast as a function!
    bspline build_spline(std::vector<double> values, double t0, double dt);

public: 

    Driver(std::vector<double> values, double t0, double dt, std::string label);
    double operator() (const double& t);
};

};
#endif