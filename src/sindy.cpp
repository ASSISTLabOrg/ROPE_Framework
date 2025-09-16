#include <iostream>
#include <iomanip>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::ublas;
using namespace boost::lambda;

typedef boost::numeric::ublas::matrix< double > mat;
typedef boost::numeric::ublas::vector< double > vec;
typedef boost::numeric::ublas::vector< int > ivec;
typedef boost::numeric::ublas::zero_matrix< double > zmat;
typedef boost::numeric::ublas::matrix< int > imat;

// typedef double (*FunctionLibrary) (vec x);

// double poly(int i, vec x) { return x(i); }
// double poly(int i, int j, vec x) { return x(i) * x(j); }
// double poly(int i, int j, int k, vec x) { return x(i) * x(j) * x(k); }

double poly(vec x, imat lib, int i)
{
    switch (lib(i,0))
    {
        case 1:
            return x(lib(i,1));
        case 2:
            return x(lib(i,1)) * x(lib(i,2));
        case 3:
            return x(lib(i,1)) * x(lib(i,2)) * x(lib(i,3));
        default:
            return 0.0;
    };
}

class sindy {

    mat m_xi; // coefficients
    imat m_lib;
    // mat m_u; // drivers

public:
    // sindy( mat xi ) : m_xi(xi) { }
    sindy (mat xi) {
        m_xi = xi;
        m_lib = imat(m_xi.size2(), 4);
    }
    // sindy(mat xi , mat u) : m_xi(xi) , m_u(u) { }

    // imat m_polylib (m_xi.size2(), 4);

    void build_func_lib( )
    {

        int n = 3;
        int k = 0;

        // O(1)
        for ( int i = 0 ; i<n ; ++i)
        {
            m_lib(i, 0) = 1;
            m_lib(i, 1) = i;
            k += 1;
        }

        // O(2)
        for (int i = 0 ; i<n ; ++i)
        {
            for (int j = i ; j<n ; ++j)
            {
                m_lib(k, 0) = 2;
                m_lib(k, 1) = i;
                m_lib(k, 2) = j;
                k += 1;
            }
        }

        cout << m_lib << endl;
    }

    void apply_func_lib( const vec &x , vec &theta)
    {
        for( int i=0 ; i<theta.size() ; ++i )
        {
            theta(i) = poly(x, m_lib, i);
        }
    }

    void operator() ( const vec &x , vec &dxdt , const double /* t */ )
    {
        vec theta( m_xi.size2() ) ; // initialize theta vector
        apply_func_lib( x , theta ) ; // fill theta vector with actual values
        dxdt = prod( m_xi, theta ) ;
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

    const double sigma = 10.0;
    const double R = 28.0;
    const double b = 8.0 / 3.0; 

    mat coeff (3, 9); // all O(1) + O(2) polynomials
    coeff(0,0) = -sigma;
    coeff(0,1) = sigma;
    coeff(1,0) = R;
    coeff(1,1) = -1.0;
    coeff(1,5) = -1.0;
    coeff(2,2) = -b;
    coeff(2,4) = 1.0;

    vec x (3) ;
    x(0) = 10.0 ;
    x(1) = 1.0 ;
    x(2) = 1.0 ; // initial conditions

    sindy problem = sindy( coeff );
    integrate( problem , x , 0.0 , 25.0 , 0.1 , write_solution );
}