#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <nlohmann/json.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::ublas;
using json = nlohmann::json;

typedef boost::numeric::ublas::matrix< double > mat;
typedef boost::numeric::ublas::vector< double > vec;
typedef std::vector<std::function<double(vec)>> fvec;

struct sindy_configuration {
    string _id;
    string model;
    int index_base;
    int n_variables;
    int n_drivers;
    std::vector<std::vector<double>> coefficients;
    std::vector<string> library;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(sindy_configuration, 
    _id, model, index_base, n_variables, n_drivers, 
    coefficients, library);

class sindy {

    const int n; // number of variables
    const int s; // number of drivers
    const int m; // number of models
    const mat xi; // coefficient matrix
    const fvec lib; // function library
    // mat m_u; // drivers

    int count_model_terms( sindy_configuration config )
    {
        int m = 0;
        for(const string &term : config.library)
        {
            int delim = term.find("_");
            string fkind = term.substr(0, delim);

            if ( fkind == "poly" )
            {
                int k = stoi(term.substr(delim + 1, 1));
                m += boost::math::binomial_coefficient<double>((n + s) + k - 1, k);
            } 

            else if ( fkind == "sin" || fkind == "cos" || fkind == "tan" )
            {
                m += (n + s);
            } 

            else 
            {
                m += 0;
            }
        }
        return m;
    }

    mat load_coeffs( sindy_configuration config )
    {
        mat coeff (n, m);
        for(const std::vector<double> ijv : config.coefficients)
        {
            coeff(ijv[0], ijv[1]) = ijv[2];
        }
        return coeff;
    }


public:

    // Initializiation
    sindy ( sindy_configuration config ) :
        n(config.n_variables),
        s(config.n_drivers),
        m(count_model_terms(config)),
        lib(load_library(config)),
        xi(load_coeffs(config)) { }
    
    fvec load_library( sindy_configuration config )
    {
        fvec funcs(m);
        int k = 0;
        
        for (int i=0 ; i<n ; ++i)
        {
            funcs[k] = [i]( vec x ){ return poly(x(i)); };
            k += 1;
        }

        for (int i=0 ; i<n ; ++i)
        {
            for (int j=i ; j < n ; ++j)
            {
                funcs[k] = [i, j]( vec x ){ return poly(x(i), x(j)); };
                k += 1;
            }
        }

        return funcs;
    }

    static double poly( double x ){ return x; }
    static double poly( double x, double y ){ return x * y; }

    void apply_func_lib( const vec x , vec &theta)
    {
        for( int i=0 ; i<theta.size() ; ++i )
        {
            theta(i) = lib[i](x);
        }
    }

    void operator() ( const vec &x , vec &dxdt , const double /* t */ )
    {
        vec theta( xi.size2() ); // initialize theta vector
        apply_func_lib( x , theta ); // fill theta vector with actual values
        dxdt = prod( xi, theta );
    }

};

void write_solution( const vec &x , const double t )
{
    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    cout << t;
    for( int i=0 ; i<x.size() ; ++i )
    {
        cout << '\t' << x(i);
    }
    cout << '\n';
}


int main(int argc, char **argv)
{
    std::ifstream f( "lorentz.json" );
    json jconfig = json::parse(f);
    const auto config = jconfig.template get<sindy_configuration>();
    const sindy problem = sindy( config );

    vec x (3) ;
    x(0) = 10.0 ;
    x(1) = 1.0 ;
    x(2) = 1.0 ; // initial conditions

    integrate( problem , x , 0.0 , 25.0 , 0.1 , write_solution );
}