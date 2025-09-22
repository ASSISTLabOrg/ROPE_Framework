#ifndef FEATURES_H   
#define FEATURES_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <boost/numeric/ublas/vector.hpp>
#include <numeric> 
#include <functional>
#include <map>
#include <boost/numeric/ublas/vector.hpp>

namespace features{

typedef boost::numeric::ublas::vector<double> vec;
const double PI = 3.1415926;

template<typename Type>
auto product(const std::vector<Type>& vec, Type init)
{
    return std::accumulate(vec.cbegin(), vec.cend(), init, std::multiplies<Type>{});
}

class Feature{
public:
    virtual double get_value(vec x);
    virtual void get_name();
};

class ConstantFeature : public Feature {
public:
    ConstantFeature();
    double get_value(vec x);
    void get_name();
};

class LinearFeature : public Feature{
public:
    const int i;
    LinearFeature(int i);
    double get_value(vec x);
    void get_name();
};

class QuadraticFeature : public Feature
{
public:
    const int i;
    const int j;
    QuadraticFeature(int i, int j);
    double get_value(vec x);
    void get_name();
};

class CubicFeature : public Feature
{
public:
    const int i;
    const int j;
    const int k;
    CubicFeature(int i, int j, int k);
    double get_value(vec x);
    void get_name();
};

class HighOrderPolynomialFeature : public Feature
{
public:
    std::vector<double> _x;
    static std::vector<double> init_subvec(std::vector<int> idx);
    void get_components(vec x);
    const std::vector<int> idx;
    HighOrderPolynomialFeature(std::vector<int> idx);
    double get_value(vec x);
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
    double get_value(vec x);
    void get_name();
};

class ExponentialFeature : public Feature
{
public:
    const int i;
    const double scale;
    ExponentialFeature(int i, double scale);
    double get_value(vec x);
    void get_name();
};

class Library
{
    public:

        std::unordered_map<std::string, int> feature_map = {
        {"constant", 0}, // 1
        {"poly_1", 1}, // x_i
        {"poly_2", 2}, // x_i * x_j
        {"poly_3", 3}, // x_i * x_j * x_k
        {"poly_N", 4}, // x_i * ... * x_n ; n > 3
        {"sin_1", 5}, // sin(a * x + b)
        {"cos_1", 6}, // cos(a * x + b)
        {"tan_1", 7}, // tan(a * x + b)
        {"exp_1", 8} // exp(a * x)
        };

        // attributes
        // const unordered_map<string, int> feature_map; // mapping used for building the library
        std::vector<Feature*> features; // vector of all features
        double m; // number of model features
        const double N; // number of variables + drivers. Assumes an enhanced state vector z = (x_0, x_1 ... x_n, u_0, u_1 ... u_s)

        // add all features based on input
        std::vector<Feature*> build_features(std::vector<std::string> feature_list, double N);

        // add singular feature to inplace feature vector
        void add_feature(std::vector<Feature*> &lib, Feature* new_feature);

        // constructors
        Library(int N); // empty library
        Library(std::vector<std::string> feature_list, int N); // build library

        // add new feature to library
        void add_feature(Feature* new_feature);

        // get value from one feature
        double get_value(vec z, Feature* feature);

        // get values, create new vector
        vec get_values(vec z);

        // get values, apply in place
        void get_values(vec &theta, vec z);

        // get name/label of a feature
        void get_name(Feature* feature);

        // returns all feature names
        void get_names();
};

}

#endif