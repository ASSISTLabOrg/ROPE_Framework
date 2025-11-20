#include "sindy.h"

const double PI = 3.1415;

namespace sindy{

/* ======================================================== Virtual Feature ======================================================== */
double Feature::get(const double& /* t */) { return 1.0; }
double Feature::get(const int& /* i */) { return 1.0; }

/* ======================================================== Linear Feature ======================================================== */
LinearFeature::LinearFeature(var_ptr var) : 
    _var0(var), _delta_time_index(0) {}

LinearFeature::LinearFeature(var_ptr var, int delta_time_index) : 
    _var0(var), _delta_time_index(delta_time_index) {}

double LinearFeature::get(const int& index) {
    return _var0 -> get(index - delta_time_index);
}

/* ======================================================== Quadratic Feature ======================================================== */

QuadraticFeature::QuadraticFeature(var_ptr var0, var_ptr var1) : 
    _var0(var0), _var1(var1), _delta_time_index(0) {}

QuadraticFeature::QuadraticFeature(var_ptr var0, var_ptr var1, int delta_time_index) : 
    _var0(var0), _var1(var1), _delta_time_index(delta_time_index) {}

double QuadraticFeature::get(const int& index) {
    double x0 = _var0 -> get(index - _delta_time_index);
    double x1 = _var1 -> get(index - _delta_time_index);
    return x0 * x1;
}

/* ======================================================== Feature Library ======================================================== */

FeatureLibrary::FeatureLibrary(feature_vector features) : _feature(features) {}

std::vector<double> FeatureLibrary::get(const int& index) {
    std::vector<double> result;
    for (feature_ptr feature : _features)
    {
        result.push_back(feature -> get(index));
    }
    return result;
}

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

// SineFeature
// SineFeature::SineFeature(int i) : _i(i) {}
// double SineFeature::get_value(const state_vector& q) {return std::sin(PI * q[_i]);}
// void SineFeature::get_name() {std::cout << "sin(z_" << i << ")";}

// CosineFeature::CosineFeature(int i) : i(i) {}
// double CosineFeature::get_value(const state_vector& z) {return std::cos(PI * z(i));}
// void CosineFeature::get_name() {std::cout << "cos(z_" << i << ")";}

// // ExponentialFeature
// ExponentialFeature::ExponentialFeature(int i) : i(i) {}
// double ExponentialFeature::get_value(const state_vector& z) {return std::exp(z(i));}
// void ExponentialFeature::get_name() {std::cout << "exp(z_" << i << ")";}

// Library
// void Library::build_features(std::vector<std::string> feature_list, double N)
// {
//     // std::vector<Feature*> lib;
//     for (const auto &term : feature_list)
//     {
//         switch (feature_map.at(term))
//         {
//             case 0: // constant
//                 add_feature(new ConstantFeature());
//                 break;

//             case 1: // Linear terms
//                 for (int i=0 ; i<N ; ++i)
//                 {
//                     add_feature(new LinearFeature(i));
//                 }
//                 break;

//             case 2: // Quadratic terms
//                 // for (int i=0 ; i<N ; ++i)
//                 // {
//                 //     for (int j=i ; j<N ; ++j)
//                 //     {
//                 //         add_feature(new QuadraticFeature(i, j));
//                 //     }
//                 // }
//                 break;

//             case 3: // Cubic terms
//                 // for (int i=0 ; i<N ; ++i)
//                 // {
//                 //     for (int j=i ; j<N ; ++j)
//                 //     {
//                 //         for (int k = j ; k<N ; ++k)
//                 //             add_feature(new CubicFeature(i, j, k));
//                 //     }
//                 // }
//                 break;

//             case 4: // N > 3 Polynomial terms
//             case 5: // sin(x)
//             case 6: // cos(x)
//             case 7: // tan(x)
//             default:
//                 break;
//         }
//     }
//     // return lib;
// }

// void Library::add_feature(std::vector<Feature*> &lib, Feature* new_feature)
// {
//     lib.push_back(new_feature);
//     m += 1;
// }

// // constructors
// Library::Library(){}
// Library::Library(std::vector<std::string> feature_list, int N)
//     {
//         N = N;
//         build_features(feature_list, N);
//     }

// // add new feature to library
// void Library::add_feature(Feature* new_feature)
// {
//     features.push_back(new_feature);
//     m += 1;
// }

// // get value from one feature
// double Library::get_value(const double&t, const state_vector& z, const driver_vector& u, Feature* feature);
// {
//     return feature -> get_value(t, z, u);
// }

// // get values, create new vector
// state_vector Library::get_values(const state_vector &z)
// {
//     state_vector result(m);
//     int i = 0;
//     for(Feature* &feature : features)
//     {
//         result(i) = get_value(z, feature);
//         ++i;
//     }
//     return result;
// }

// // get values, apply in place
// void Library::get_values(state_vector &theta, const state_vector &z)
// {
//     int i = 0;
//     for(Feature* &feature : features)
//     {
//         theta(i) = get_value(z, feature);
//         ++i;
//     }
// }

// void Library::get_name(Feature* feature)
// {
//     feature -> get_name();
// }

// void Library::get_names()
// {
//     for (Feature* &feature : features)
//     {
//         get_name(feature);
//         std::cout << std::endl;
//     }
// }

}

// // get value from one feature
// double get_value(const double&t, const state_vector& z, const driver_vector& u, Feature* feature);

// // get values, create new vector
// vec get_values(const double& t, const state_vector& z, const driver_vector& u);

// // get values, apply in place
// void get_values(const double& t, state_vector &theta, const state_vector& z, const driver_vector& u);
