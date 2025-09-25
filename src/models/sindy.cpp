#include "features.h"
#include "sindy.h"

using features::Library;
using boost::numeric::ublas::prod;
using boost::numeric::odeint::integrate;

namespace sindy
{

    void write_solution( const vec &x , const double t )
    {
        std::cout << std::fixed;
        std::cout << std::setprecision(2);
    
        std::cout << t;
        for( int i=0 ; i<x.size() ; ++i )
        {
            std::cout << '\t' << x(i);
        }
        std::cout << '\n';
    }

    void SINDYc::push(const vec &x , vec &dxdt , double /* t */)
    {
        dxdt = prod(A, library.get_values(x));
    }

    SINDYc::SINDYc(std::string config_file){}

    SINDYc::SINDYc(double n, double s, std::vector<std::string> features, std::vector<std::vector<double>> coeffs)
    {
        n_variables = n;
        n_controls = s;
        library.build_features(features, n + s);
        n_features = library.m;
        A = build_coeffs(coeffs);
    }

    mat SINDYc::build_coeffs(std::vector<std::vector<double>> coeffs)
    {   
        mat output(n_variables + n_controls, n_features);
        for(const std::vector<double> ijv : coeffs)
        {
            output(ijv[0], ijv[1]) = ijv[2];
        }
        return output;
    }

    void SINDYc::solve(vec& initial_state, double t_start, double t_end, double dt)
    {
        auto lambda_system = [this](const vec& x, vec& dxdt, const double t) 
        {
            this->push(x, dxdt, t);
        };

        integrate(lambda_system, initial_state, t_start, t_end, dt, write_solution);
    }

};