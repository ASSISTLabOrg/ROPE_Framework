#include "../src/models/features.h"
#include "../src/models/models.h"

using namespace features;
using namespace models;
using boost::numeric::odeint::integrate;
// using features::Library;
// using utils::vec;

int main(int argc, char **argv)
{ 
    vec x (3) ;
    x(0) = 10.0 ;
    x(1) = 1.0 ;
    x(2) = 1.0 ; // initial conditions

    double t0 = 0.0;
    double t1 = 25.0;
    double dt = 0.1;

    std::vector<std::string> feature_list = {"poly_1", "poly_2"};
    // Library library = Library(feature_list, x.size());

    // vec test = library.get_values(x);

    // for (int i=0 ; i<library.m ; ++i)
    // {
    //     std::cout << "The value of ";
    //     library.features[i] -> get_name();
    //     std::cout << " is: " << test(i) << std::endl;
    // }

    std::vector<std::vector<double>> coeffs_ijv = 
        {
        {0,0,-10.0}, {0,1,10.0}, {1,0,28.0}, 
        {1,1,-1.0}, {1,5,-1.0}, {2,2,-2.6667}, 
        {2,4,1.0}
        };

    // CoupledODESystem system = CoupledODESystem(3, 0, library, coeffs_ijv);
    SINDYc sindy = SINDYc(3, 0, feature_list, coeffs_ijv);
    
    // sindy.library.get_names();
    sindy.list_features();
    sindy.predict(x, t0, t1, dt);
    // integrate(system, x, t0, t1, dt, write_solution);

    // std::cout << feature.get_value(x) << std::endl;
    return 0;
}