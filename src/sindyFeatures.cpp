#include "sindyFeatures.h"

const double PI = 3.1415;

namespace sindyFeatures{

/* ======================================================== Virtual Feature ======================================================== */
double Feature::get_value(const double& /*t*/) {
    return 1.0;
}

/* ======================================================== Linear Feature ======================================================== */
LinearFeature::LinearFeature(int i) : _i(i) {}

double LinearFeature::get_value(const state_vector& q) {
    return q[_i];
}

/* ======================================================== Quadratic Feature ======================================================== */

QuadraticFeature::QuadraticFeature(int i, int j) : _i(i), _j(j) {}

double QuadraticFeature::get_value(const state_vector& q) {
    return q[_i] * q[_j];
}

/* ======================================================== Cubic Feature ======================================================== */
CubicFeature::CubicFeature(int i, int j, int k) : _i(i), _j(j), _k(k) {}

double CubicFeature::get_value(const state_vector& q){
    return q[_i] * q[_j] * q[_k]
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
