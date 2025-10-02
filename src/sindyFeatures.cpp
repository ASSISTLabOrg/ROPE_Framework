#include "sindyFeatures.h"

const double PI = 3.1415;

namespace sindyFeatures{

/* ======================================================== Virtual Feature ======================================================== */
double Feature::get_value(const double& /*t*/, const state_vector& /*z*/, const driver_vector& /*u*/) {
    return 1.0;
}

void Feature::get_name() {
    std::cout << "Null";
}

/* ======================================================== Linear Feature ======================================================== */
LinearFeature::LinearFeature(int i) : i(i), i_is_state(true) {}
LinearFeature::LinearFeature(int i, bool i_is_state) : i(i), i_is_state(i_is_state) {}

double LinearFeature::get_value(const state_vector& z) {
    return z(i);
}

double LinearFeature::get_value(const double& t, const state_vector& z, const driver_vector& u) {
    ( is_state ) 
        ? return z(i) 
        : return u[i](t);
}

void LinearFeature::get_name() {
    ( is_state ) 
        ? std::cout << "z" << i;
        : std::cout << "u" << i;
}

struct Index { const int value; };
struct DriverIndex : public Index { const int value; };
struct StateIndex : public Index { const int value; };

double get(const double& t, const state_vector& z, const driver_vector& u, const DriverIndex& i) { return u[i.value](t); }
double get(const double& t, const state_vector& z, const driver_vector& u, const StateIndex& i) { return z(i.value); }


/* ======================================================== Quadratic Feature ======================================================== */
QuadraticFeature::QuadraticFeature(Index* i, Index* j) : i(i), j(j) {}

double QuadraticFeature::get_value(const double& t, const state_vector& z, const driver_vector& u) {
    return get(t, z, u, i) * get(t, z, u, j);
}

void QuadraticFeature::get_name() {
    ( i_is_state )
        ? std::string istr = "z_";
        : std::string istr = "u_";
    ( j_is_state )
        ? std::string jstr = "z_";
        : std::string jstr = "u_";
    std::cout << istr << i << "*" << jstr << j;
}

/* ======================================================== Cubic Feature ======================================================== */
CubicFeature::CubicFeature(int i, int j, int k) : 
    i(i), j(j), k(k), i_is_state(true), j_is_state(true), k_is_state(true) {}

CubicFeature::CubicFeature(int i, int j, int k, bool i_is_state, bool j_is_state, bool k_is_state) : 
    i(i), j(j), k(k), i_is_state(i_is_state){}


double CubicFeature::get_value(const double&t, const state_vector& z, const driver_vector& u) {
    if ( jth_is_state ){
        return z(i) * z(j) * u[k](t);
    } else {
        return z(i) * u[j](t) * u[k](t);
    }
}

double CubicFeature::get_value(const double& t, const driver_vector& u){
    return u[i](t) * u[j](t) * u[k](t);
}

void CubicFeature::get_name() {std::cout << "z" << i << "z" << j << "z" << k;}

// // HighOrderPolynomialFeature
// std::vector<double> HighOrderPolynomialFeature::init_subvec(std::vector<int> idx)
// {
//     std::vector<double> _z(idx.size());
//     return _z;
// }
// void HighOrderPolynomialFeature::get_components(const state_vector& z)
// {
//     for(int i=0 ; i < _z.size() ; ++i)
//     {
//         _z[i] = z(idx[i]);
//     }
// }
// HighOrderPolynomialFeature::HighOrderPolynomialFeature(std::vector<int> idx) : idx(idx), _z(init_subvec(idx)) {}
// double HighOrderPolynomialFeature::get_value(const state_vector& z) 
// {
//     HighOrderPolynomialFeature::get_components(z);
//     return det_diag(_z, 1.0);
// }
// void HighOrderPolynomialFeature::get_name() 
// {
//     for (const auto &i : idx)
//     {
//         std::cout << "z" << i;
//     }
// }

// // SineFeature
// SineFeature::SineFeature(int i) : i(i) {}
// double SineFeature::get_value(const state_vector& z) {return std::sin(PI * z(i));}
// void SineFeature::get_name() {std::cout << "sin(z_" << i << ")";}

// CosineFeature::CosineFeature(int i) : i(i) {}
// double CosineFeature::get_value(const state_vector& z) {return std::cos(PI * z(i));}
// void CosineFeature::get_name() {std::cout << "cos(z_" << i << ")";}

// // ExponentialFeature
// ExponentialFeature::ExponentialFeature(int i) : i(i) {}
// double ExponentialFeature::get_value(const state_vector& z) {return std::exp(z(i));}
// void ExponentialFeature::get_name() {std::cout << "exp(z_" << i << ")";}

// Library
void Library::build_features(std::vector<std::string> feature_list, double N)
{
    // std::vector<Feature*> lib;
    for (const auto &term : feature_list)
    {
        switch (feature_map.at(term))
        {
            case 0: // constant
                add_feature(new ConstantFeature());
                break;

            case 1: // Linear terms
                for (int i=0 ; i<N ; ++i)
                {
                    add_feature(new LinearFeature(i));
                }
                break;

            case 2: // Quadratic terms
                for (int i=0 ; i<N ; ++i)
                {
                    for (int j=i ; j<N ; ++j)
                    {
                        add_feature(new QuadraticFeature(i, j));
                    }
                }
                break;

            case 3: // Cubic terms
                for (int i=0 ; i<N ; ++i)
                {
                    for (int j=i ; j<N ; ++j)
                    {
                        for (int k = j ; k<N ; ++k)
                            add_feature(new CubicFeature(i, j, k));
                    }
                }
                break;

            case 4: // N > 3 Polynomial terms
            case 5: // sin(x)
            case 6: // cos(x)
            case 7: // tan(x)
            default:
                break;
        }
    }
    // return lib;
}

void Library::add_feature(std::vector<Feature*> &lib, Feature* new_feature)
{
    lib.push_back(new_feature);
    m += 1;
}

// constructors
Library::Library(){}
Library::Library(std::vector<std::string> feature_list, int N)
    {
        N = N;
        build_features(feature_list, N);
    }

// add new feature to library
void Library::add_feature(Feature* new_feature)
{
    features.push_back(new_feature);
    m += 1;
}

// get value from one feature
double Library::get_value(const double&t, const state_vector& z, const driver_vector& u, Feature* feature);
{
    return feature -> get_value(t, z, u);
}

// get values, create new vector
state_vector Library::get_values(const state_vector &z)
{
    state_vector result(m);
    int i = 0;
    for(Feature* &feature : features)
    {
        result(i) = get_value(z, feature);
        ++i;
    }
    return result;
}

// get values, apply in place
void Library::get_values(state_vector &theta, const state_vector &z)
{
    int i = 0;
    for(Feature* &feature : features)
    {
        theta(i) = get_value(z, feature);
        ++i;
    }
}

void Library::get_name(Feature* feature)
{
    feature -> get_name();
}

void Library::get_names()
{
    for (Feature* &feature : features)
    {
        get_name(feature);
        std::cout << std::endl;
    }
}

}

// // get value from one feature
// double get_value(const double&t, const state_vector& z, const driver_vector& u, Feature* feature);

// // get values, create new vector
// vec get_values(const double& t, const state_vector& z, const driver_vector& u);

// // get values, apply in place
// void get_values(const double& t, state_vector &theta, const state_vector& z, const driver_vector& u);
