#include "features.h"

const double PI = 3.1415;

namespace features{

// Feature
double Feature::get_value(const vec& z) {return 0.0;}
void Feature::get_name() {std::cout << "Null";}

// Constant Feature
double ConstantFeature::get_value(const vec& z) {return 1.0;}
void ConstantFeature::get_name() {std::cout << "Constant";}

// LinearFeature
LinearFeature::LinearFeature(int i) : i(i) {}
double LinearFeature::get_value(const vec& z) {return z(i);}
void LinearFeature::get_name() {std::cout << "z" << i;}

// QuadraticFeature
QuadraticFeature::QuadraticFeature(int i, int j) : i(i), j(j) {}
double QuadraticFeature::get_value(const vec& z) {return z(i) * z(j);}
void QuadraticFeature::get_name() {std::cout << "z" << i << "z" << j;}

// CubicFeature
CubicFeature::CubicFeature(int i, int j, int k) : i(i), j(j), k(k) {}
double CubicFeature::get_value(const vec& z) {return z(i) * z(j) * z(k);}
void CubicFeature::get_name() {std::cout << "z" << i << "z" << j << "z" << k;}

// HighOrderPolynomialFeature
std::vector<double> HighOrderPolynomialFeature::init_subvec(std::vector<int> idx)
{
    std::vector<double> _z(idx.size());
    return _z;
}
void HighOrderPolynomialFeature::get_components(const vec& z)
{
    for(int i=0 ; i < _z.size() ; ++i)
    {
        _z[i] = z(idx[i]);
    }
}
HighOrderPolynomialFeature::HighOrderPolynomialFeature(std::vector<int> idx) : idx(idx), _z(init_subvec(idx)) {}
double HighOrderPolynomialFeature::get_value(const vec& z) 
{
    HighOrderPolynomialFeature::get_components(z);
    return det_diag(_z, 1.0);
}
void HighOrderPolynomialFeature::get_name() 
{
    for (const auto &i : idx)
    {
        std::cout << "z" << i;
    }
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
double SinusoidalFeature::get_value(const vec& z) {return std::sin(scale * z(i) + phase);}
void SinusoidalFeature::get_name() {std::cout << "sin(" << scale << " * " << "z" << i << " + " << phase << ")";}

// ExponentialFeature
ExponentialFeature::ExponentialFeature(int i, double scale) : i(i), scale(scale) {}
double ExponentialFeature::get_value(const vec& z) {return std::exp(scale * z(i));}
void ExponentialFeature::get_name() {std::cout << "exp(" << scale << " * " << "z" << i << ")";}

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
double Library::get_value(const vec& z, Feature* feature)
{
    return feature -> get_value(z);
}

// get values, create new vector
vec Library::get_values(const vec &z)
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
void Library::get_values(vec &theta, const vec &z)
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