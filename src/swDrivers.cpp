namespace swDrivers{

    typedef boost::math::interpolators::cardinal_cubic_b_spline<double> bspline;
    
    class swDriver{
    
        // std::string label = "Null";
        const bspline driver_function; // data, cast as a function!
    
        Driver(std::string data_file) : driver_function(build_spline(data_file)) {}
        Driver(std::vector<double> values, double t0, double dt) : func(build_spline(values, t0, dt)) {}
    
        bspline build_spline(std::string data_file)
        {
            // read data file
            data = ...;
            bspline d(data.values, data.values.size(), data.t0, data.dt);
            return d;
        }
    
        bspline build_spline(std::vector<double> values, double t0, double dt)
        {
            bspline d(values, values.size(), t0, dt);
            return d;
        }
    
        double operator() (double t)
        {
            return driver_function(t);
        }
    
        double operator() (std::vector<double> times)
        {
            std::vector<double> x;
            for (const double& t : times)
            {
                x.push_back(driver_function(t));
            }
        }
    
    };
    
    struct swDrivers{
    
        const std::vector<swDriver> drivers;
    
        double operator() (const double& t, const double& i)
        {
            return drivers[i](t);
        }
    
        std::vector<double> operator() (double t)
        {
            std::vector<double> x;
            for (const swDriver& driver : drivers)
            {
                x.push_back(driver(t));
            }
        }
    
    }
};