#ifndef ROPE_MODELS_SINDYFEATURES_H   
#define ROPE_MODELS_SINDYFEATURES_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <numeric> 
#include <functional>
#include <map>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace sindyFeatures{ 

typedef boost::numeric::ublas::vector<double> state_vector;
typedef boost::numeric::ublas::matrix<double> matrix;
typedef std::vector<Feature*> feature_vector;

template<typename Type>
auto det_diag(const std::vector<Type>& v, Type init)
{
    return std::accumulate(v.cbegin(), v.cend(), init, std::multiplies<Type>{});
}

class Feature{
public:
    virtual double get_value(const double t, const state_vector& z, const driver_vector& u);
    virtual void get_name();
};

class LinearFeature : public Feature{
public:
    const int i;
    LinearFeature(int i);
    double get_value(const state_vector& z);
    double get_value(const double t, const state_vector& z, const driver_vector& u);
    void get_name();
};

double get(const state_vector& z){ return z(i); }
double get(const double& t, const driver_vector& u){ return u[i](t); }

class QuadraticFeature : public Feature
{
public:
    const int i;
    const int j;
    QuadraticFeature(int i, int j);
    QuadraticFeature(int i, int j, bool i_is_state, bool j_is_state);
    double get_value(const state_vector& z);
    double get_value(const double t, const state_vector& z, const driver_vector& u);
    void get_name();
};

class CubicFeature : public Feature
{
public:
    const int i;
    const int j;
    const int k;
    CubicFeature(int i, int j, int k);
    double get_value(const state_vector& z);
    double get_value(const double t, const state_vector& z, const driver_vector& u);
    void get_name();
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
double get_value(const double t, const state_vector& u);

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

class Library
{
    public:

        std::unordered_map<std::string, int> feature_map = {
        // {"constant", 0}, // 1
        {"poly_1", 1}, // z_i
        {"poly_2", 2}, // z_i * z_j
        {"poly_3", 3}, // z_i * z_j * z_k
        // {"poly_N", 4}, // z_i * ... * z_n ; n > 3
        // {"sin_1", 5}, // sin(a * z_i + b)
        // {"cos_1", 6}, // cos(a * z_i + b)
        // {"tan_1", 7}, // tan(a * z_i + b)
        // {"exp_1", 8} // exp(a * z_i)
        };

        // attributes
        // const unordered_map<string, int> feature_map; // mapping used for building the library
        feature_vector features = {}; // vector of all features
        double m = 0; // number of model features
        double N = 0; // number of variables + drivers. Assumes an enhanced state vector z = (x_0, x_1 ... x_n, u_0, u_1 ... u_s)

        // add all features based on input
        // std::vector<Feature*> build_features(std::vector<std::string> feature_list, double N);
        void build_features(std::vector<std::string> feature_list, double N);

        // add singular feature to inplace feature vector
        void add_feature(feature_vector &lib, Feature* new_feature);

        // constructors
        Library();
        Library(std::vector<std::string> feature_list, int N); // build library

        // add new feature to library
        void add_feature(Feature* new_feature);

        // get value from one feature
        double get_value(const double& t, const state_vector& z, const driver_vector& u, Feature* feature);

        // get values, create new vector
        vec get_values(const double& t, const state_vector& z, const driver_vector& u);

        // get values, apply in place
        void get_values(const double& t, state_vector &theta, const state_vector& z, const driver_vector& u);

        // get name/label of a feature
        void get_name(Feature* feature);

        // returns all feature names
        void get_names();
};

}

#endif