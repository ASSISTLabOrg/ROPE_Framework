#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using Eigen::MatrixXd;

typedef std::vector< double > state_type;

const double sigma = 10.0;
const double R = 28.0;
const double b = 8.0 / 3.0;

/* The rhs of x' = f(x) defined as a class */
class sindy {

    MatrixXd xi;

public:
    sindy( MatrixXd xi ) : m_xi(xi) { }

    void operator() ( const state_type &x , state_type &dxdt , const double /* t */ )
    {
        state_type theta = {x[0], x[1], x[2], x[0]*x[1], x[0]*x[2], x[1]*x[2]};
        dxdt = xi * theta;
    }
};

void write_lorenz( const state_type &x , const double t )
{
    cout << t << '\t' << x[0] << '\t' << x[1] << '\t' << x[2] << endl;
}


int main(int argc, char **argv)
{
    MatrixXd coeff = MatrixXd::Zero(3,6);
    coeff(0,0) = -sigma;
    coeff(0,1) = sigma;
    coeff(1,0) = R;
    coeff(1,1) = -1.0;
    coeff(1,4) = -1.0;
    coeff(2,2) = -b;
    coeff(2,3) = 1.0;

    std::cout << coeff << "\n";

    state_type x = { 10.0 , 1.0 , 1.0 }; // initial conditions
    integrate( sindy(coeff) , x , 0.0 , 25.0 , 0.1 , write_lorenz );
}