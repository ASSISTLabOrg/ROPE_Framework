#ifndef ROPE_SINDY_H   
#define ROPE_SINDY_H

#include <memory>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <numeric> 
#include <functional>
#include <map>
#include "variables.h"

namespace sindy{

using variables::variable_ptr;

template<typename Type>
auto det_diag(const std::vector<Type>& v, Type init)
{
    return std::accumulate(v.cbegin(), v.cend(), init, std::multiplies<Type>{});
}

struct Feature{
    virtual double get(const double& /* t */);
    virtual double get(const int& /* i */);
};

typedef std::unique_ptr<Feature> feature_ptr;
typedef std::vector<feature_ptr> feature_vector;

struct LinearFeature : Feature 
{
    variable_ptr _var0;
    const int _delta_time_index;
    LinearFeature(variable_ptr var);
    LinearFeature(variable_ptr var, int delta_time_index);
    // double get(const double& time);
    double get(const int& index);
};

struct QuadraticFeature : Feature
{
    variable_ptr _var0;
    variable_ptr _var1;
    const int _delta_time_index;
    QuadraticFeature(variable_ptr var0, variable_ptr var1, int delta_time_index);
    double get(const int& index);
};

struct FeatureLibrary
{
    feature_vector _features;
    FeatureLibrary(feature_vector features);
    std::vector<double> get(const int& index);
};

// class HighOrderPolynomialFeature : public Feature
// {
// public:
//     std::vector<double> _z;
//     static std::vector<double> init_subvec(std::vector<int> idx);
//     void get_components(const state_vector& z);
//     const std::vector<int> idx;
//     HighOrderPolynomialFeature(std::vector<int> idx);
//     double get_value(const state_vector& z);
//     void get_name();
// };

// class SineFeature : public Feature
// {
// public:
//     const int i;
//     SineFeature(int i);
//     double get_value(const state_vector& z);
//     void get_name();
// };

// class CosineFeature : public Feature
// {
// public:
//     const int i;
//     CosineFeature(int i);
//     double get_value(const state_vector& z);
//     void get_name();
// };

// class ExponentialFeature : public Feature
// {
// public:
//     const int i;
//     ExponentialFeature(int i);
//     double get_value(const state_vector& z);
//     void get_name();
// };

// class Library
// {
//     public:

//         std::unordered_map<std::string, int> feature_map = {
//         // {"constant", 0}, // 1
//         {"poly_1", 1}, // z_i
//         {"poly_2", 2}, // z_i * z_j
//         {"poly_3", 3}, // z_i * z_j * z_k
//         // {"poly_N", 4}, // z_i * ... * z_n ; n > 3
//         // {"sin_1", 5}, // sin(a * z_i + b)
//         // {"cos_1", 6}, // cos(a * z_i + b)
//         // {"tan_1", 7}, // tan(a * z_i + b)
//         // {"exp_1", 8} // exp(a * z_i)
//         };

//         // attributes
//         // const unordered_map<string, int> feature_map; // mapping used for building the library
//         feature_vector features = {}; // vector of all features
//         double m = 0; // number of model features
//         double N = 0; // number of variables + drivers. Assumes an enhanced state vector z = (x_0, x_1 ... x_n, u_0, u_1 ... u_s)

//         // add all features based on input
//         // std::vector<Feature*> build_features(std::vector<std::string> feature_list, double N);
//         void build_features(std::vector<std::string> feature_list, double N);

//         // add singular feature to inplace feature vector
//         void add_feature(feature_vector &lib, Feature* new_feature);

//         // constructors
//         Library();
//         Library(std::vector<std::string> feature_list, int N); // build library

//         // add new feature to library
//         void add_feature(Feature* new_feature);

//         // get value from one feature
//         double get_value(const double& t, const state_vector& z, const driver_vector& u, Feature* feature);

//         // get values, create new vector
//         vec get_values(const double& t, const state_vector& z, const driver_vector& u);

//         // get values, apply in place
//         void get_values(const double& t, state_vector &theta, const state_vector& z, const driver_vector& u);

//         // get name/label of a feature
//         void get_name(Feature* feature);

//         // returns all feature names
//         void get_names();
// };

}

#endif