#include <spaceWeatherDrivers.h>

namespace spaceWeatherDrivers{

    typedef boost::math::interpolators::cardinal_cubic_b_spline<double> bspline;
    typedef std::vector<SpaceWeatherDriver> driver_vector;
    
    class SpaceWeatherDriver{
    
        // std::string label = "Null";
        const bspline driver_function; // data, cast as a function!
    
        // Driver(std::string data_file) : driver_function(build_spline(data_file)) {}
        Driver(std::vector<double> values, double t0, double dt) : func(build_spline(values, t0, dt)) {}
    
        // bspline build_spline(std::string data_file)
        // {
            // read data file
            // data = ...
            // bspline d(data.values, data.values.size(), data.t0, data.dt);
            // return d;
        // }
    
        bspline build_spline(std::vector<double> values, double t0, double dt)
        {
            bspline d(values, values.size(), t0, dt);
            return d;
        }
    
        double operator() (const double& t)
        {
            return driver_function(t);
        }
    
        // std::vector<double> operator() (std::vector<double> times)
        // {
        //     std::vector<double> x;
        //     for (const double& t : times)
        //     {
        //         x.push_back(driver_function(t));
        //     }
        //     return x;
        // }
    
    };
};