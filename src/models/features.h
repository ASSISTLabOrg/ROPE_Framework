#ifndef FEATURES_H   
#define FEATURES_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <numeric> 
#include <functional>
#include <map>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

typedef boost::numeric::ublas::vector<double> vec;
typedef boost::numeric::ublas::matrix<double> mat;

namespace features{ 

template<typename Type>
auto det_diag(const std::vector<Type>& v, Type init)
{
    return std::accumulate(v.cbegin(), v.cend(), init, std::multiplies<Type>{});
}

class Feature{
public:
    virtual double get_value(const vec& z);
    virtual void get_name();
};

class ConstantFeature : public Feature {
public:
    double get_value(const vec& z);
    void get_name();
};

class LinearFeature : public Feature{
public:
    const int i;
    LinearFeature(int i);
    double get_value(const vec& z);
    void get_name();
};

class QuadraticFeature : public Feature
{
public:
    const int i;
    const int j;
    QuadraticFeature(int i, int j);
    double get_value(const vec& z);
    void get_name();
};

class CubicFeature : public Feature
{
public:
    const int i;
    const int j;
    const int k;
    CubicFeature(int i, int j, int k);
    double get_value(const vec& z);
    void get_name();
};

class HighOrderPolynomialFeature : public Feature
{
public:
    std::vector<double> _z;
    static std::vector<double> init_subvec(std::vector<int> idx);
    void get_components(const vec& z);
    const std::vector<int> idx;
    HighOrderPolynomialFeature(std::vector<int> idx);
    double get_value(const vec& z);
    void get_name();
};

class SinusoidalFeature : public Feature
{
public:
    double get_scale(double val);
    double get_phase(double val);
    const int i;
    const double scale;
    const double phase;
    const std::string rad_or_deg;
    SinusoidalFeature(int i, double scale, double phase, std::string rad_or_deg);
    double get_value(const vec& z);
    void get_name();
};

class ExponentialFeature : public Feature
{
public:
    const int i;
    const double scale;
    ExponentialFeature(int i, double scale);
    double get_value(const vec& zx);
    void get_name();
};

class Library
{
    public:

        std::unordered_map<std::string, int> feature_map = {
        {"constant", 0}, // 1
        {"poly_1", 1}, // z_i
        {"poly_2", 2}, // z_i * z_j
        {"poly_3", 3}, // z_i * z_j * z_k
        {"poly_N", 4}, // z_i * ... * z_n ; n > 3
        {"sin_1", 5}, // sin(a * z_i + b)
        {"cos_1", 6}, // cos(a * z_i + b)
        {"tan_1", 7}, // tan(a * z_i + b)
        {"exp_1", 8} // exp(a * z_i)
        };

        // attributes
        // const unordered_map<string, int> feature_map; // mapping used for building the library
        std::vector<Feature*> features = {}; // vector of all features
        double m = 0; // number of model features
        double N = 0; // number of variables + drivers. Assumes an enhanced state vector z = (x_0, x_1 ... x_n, u_0, u_1 ... u_s)

        // add all features based on input
        // std::vector<Feature*> build_features(std::vector<std::string> feature_list, double N);
        void build_features(std::vector<std::string> feature_list, double N);

        // add singular feature to inplace feature vector
        void add_feature(std::vector<Feature*> &lib, Feature* new_feature);

        // constructors
        Library();
        Library(std::vector<std::string> feature_list, int N); // build library

        // add new feature to library
        void add_feature(Feature* new_feature);

        // get value from one feature
        double get_value(const vec& z, Feature* feature);

        // get values, create new vector
        vec get_values(const vec& z);

        // get values, apply in place
        void get_values(vec &theta, const vec& z);

        // get name/label of a feature
        void get_name(Feature* feature);

        // returns all feature names
        void get_names();
};

}

#endif