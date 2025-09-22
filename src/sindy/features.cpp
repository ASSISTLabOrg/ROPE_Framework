#include "features.h"

namespace features{

// Feature
double Feature::get_value(vec x) {return 0.0;}
double get_value(vec x) {return 0.0;}
void Feature::get_name() {std::cout << "Null" << std::endl;}

// Constant Feature
ConstantFeature::ConstantFeature(){}
double ConstantFeature::get_value(vec x) {return 1.0;}
void ConstantFeature::get_name() {std::cout << "Constant" << std::endl;}

// LinearFeature
LinearFeature::LinearFeature(int i) : i(i) {}
double LinearFeature::get_value(vec x) {return x(i);}
void LinearFeature::get_name() {std::cout << "x" << i << std::endl;}

// QuadraticFeature
QuadraticFeature::QuadraticFeature(int i, int j) : i(i), j(j) {}
double QuadraticFeature::get_value(vec x) {return x(i) * x(j);}
void QuadraticFeature::get_name() {std::cout << "x" << i << "x" << j << std::endl;}

// CubicFeature
CubicFeature::CubicFeature(int i, int j, int k) : i(i), j(j), k(k) {}
double CubicFeature::get_value(vec x) {return x(i) * x(j) * x(k);}
void CubicFeature::get_name() {std::cout << "x" << i << "x" << j << "x" << k << std::endl;}

// HighOrderPolynomialFeature
std::vector<double> HighOrderPolynomialFeature::init_subvec(std::vector<int> idx)
{
    std::vector<double> _x(idx.size());
    return _x;
}
void HighOrderPolynomialFeature::get_components(vec x)
{
    for(int i=0 ; i < _x.size() ; ++i)
    {
        _x[i] = x(idx[i]);
    }
}
HighOrderPolynomialFeature::HighOrderPolynomialFeature(std::vector<int> idx) : idx(idx), _x(init_subvec(idx)) {}
double HighOrderPolynomialFeature::get_value(vec x) 
{
    HighOrderPolynomialFeature::get_components(x);
    return product(_x, 1.0);
}
void HighOrderPolynomialFeature::get_name() 
{
    for (auto &i : idx)
    {
        std::cout << "x" << i;
    }
    std::cout << std::endl;
}

// SinusoidalFeature
double SinusoidalFeature::get_scale(double val)
{
    if (rad_or_deg == "rad")
    {
        return val;
    }
    else
    {
        return val * 2.0 * PI;
    }
}
double SinusoidalFeature::get_phase(double val)
{
    if (rad_or_deg == "rad")
    {
        return val;
    }
    else
    {
        return val * PI / 180;
    }
}
SinusoidalFeature::SinusoidalFeature(int i, double scale, double phase, std::string rad_or_deg="rad") : 
    i(i), rad_or_deg(rad_or_deg), scale(get_scale(scale)), phase(get_phase(phase))  {}
double SinusoidalFeature::get_value(vec x) {return std::sin(scale * x(i) + phase);}
void SinusoidalFeature::get_name() {std::cout << "sin(" << scale << " * " << "x" << i << " + " << phase << ")" << std::endl;}

// ExponentialFeature
ExponentialFeature::ExponentialFeature(int i, double scale) : i(i), scale(scale) {}
double ExponentialFeature::get_value(vec x) {return std::exp(scale * x(i));}
void ExponentialFeature::get_name() {std::cout << "exp(" << scale << " * " << "x" << i << ")" << std::endl;}

// Library
std::vector<Feature*> Library::build_features(std::vector<std::string> feature_list, double N)
{
    std::vector<Feature*> lib;
    for (auto &term : feature_list)
    {
        switch (feature_map.at(term))
        {
            case 0: // constant
                add_feature(lib, new ConstantFeature());
                break;

            case 1: // Linear terms
                for (int i=0 ; i<N ; ++i)
                {
                    add_feature(lib, new LinearFeature(i));
                }
                break;

            case 2: // Quadratic terms
                for (int i=0 ; i<N ; ++i)
                {
                    for (int j=i ; j<N ; ++j)
                    {
                        add_feature(lib, new QuadraticFeature(i, j));
                    }
                }
                break;

            case 3: // Cubic terms
                for (int i=0 ; i<N ; ++i)
                {
                    for (int j=i ; j<N ; ++j)
                    {
                        for (int k = j ; k<N ; ++k)
                            add_feature(lib, new CubicFeature(i, j, k));
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
    return lib;
}

void Library::add_feature(std::vector<Feature*> &lib, Feature* new_feature)
{
    lib.push_back(new_feature);
    m += 1;
}

// constructors
Library::Library(int N) : N(N) {} // empty library
Library::Library(std::vector<std::string> feature_list, int N) : 
    N(N), 
    features(build_features(feature_list, N)) 
    {}

// add new feature to library
void Library::add_feature(Feature* new_feature)
{
    features.push_back(new_feature);
    m += 1;
}

// get value from one feature
double Library::get_value(vec z, Feature* feature)
{
    return feature -> get_value(z);
}

// get values, create new vector
vec Library::get_values(vec z)
{
    vec result(m);
    int i = 0;
    for(Feature* &feature : features)
    {
        result(i) = get_value(z, feature);
        ++i;
    }
    return result;
}

// get values, apply in place
void Library::get_values(vec &theta, vec z)
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
        Library::get_name(feature);
    }
}

}