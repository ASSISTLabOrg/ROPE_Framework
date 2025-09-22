#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <boost/numeric/ublas/vector.hpp>
#include <numeric> 
#include <functional>
#include <map>
#include "features.h"

using namespace std;
typedef boost::numeric::ublas::vector< double > vec;
const double PI = 3.1415926;

template<typename Type>
auto product(const std::vector<Type>& vec, Type init)
{
    return std::accumulate(vec.cbegin(), vec.cend(), init, std::multiplies<Type>{});
}

class Feature{
public:
    virtual double get_value(vec x) {return 0.0;};
    virtual void get_name() {cout << "Null" << endl;}
};

class ConstantFeature : public Feature
{
public:
    ConstantFeature(){}
    double get_value(vec x) override {return 1.0;};
    void get_name() override {cout << "Constant" << endl;}
};

class LinearFeature : public Feature
{
public:
    const int i;
    LinearFeature(int i) : i(i) {}
    double get_value(vec x) override {return x(i);}
    void get_name() override {cout << "x" << i << endl;}
};

class QuadraticFeature : public Feature
{
public:
    const int i;
    const int j;
    QuadraticFeature(int i, int j) : i(i), j(j) {}
    double get_value(vec x) override {return x(i) * x(j);}
    void get_name() override {cout << "x" << i << "x" << j << endl;}
};

class CubicFeature : public Feature
{
public:
    const int i;
    const int j;
    const int k;
    CubicFeature(int i, int j, int k) : i(i), j(j), k(k) {}
    double get_value(vec x) override {return x(i) * x(j) * x(k);}
    void get_name() override {cout << "x" << i << "x" << j << "x" << k << endl;}
};

class HighOrderPolynomialFeature : public Feature
{
private:
    static vector<double> init_subvec(vector<int> idx)
    {
        vector<double> _x(idx.size());
        return _x;
    }

    void get_components(vec x)
    {
        for(int i=0 ; i < _x.size() ; ++i)
        {
            _x[i] = x(idx[i]);
        }
    }

public:
    const vector<int> idx;
    vector<double> _x;
    HighOrderPolynomialFeature(vector<int> idx) : idx(idx), _x(init_subvec(idx)) {}

    double get_value(vec x) override 
    {
        get_components(x);
        return product(_x, 1.0);
    }

    void get_name() override 
    {
        for (auto &i : idx)
        {
            cout << "x" << i;
        }
        cout << endl;
    }
};

class SinusoidalFeature : public Feature
{
private:
    double get_scale(double val)
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

    double get_phase(double val)
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
public:
    const int i;
    const double scale;
    const double phase;
    const string rad_or_deg;
    SinusoidalFeature(int i, double scale, double phase, string rad_or_deg="rad") : 
        i(i), rad_or_deg(rad_or_deg), scale(get_scale(scale)), phase(get_phase(phase))  {}


    double get_value(vec x) override {return sin(scale * x(i) + phase);}
    void get_name() override {cout << "sin(" << scale << " * " << "x" << i << " + " << phase << ")";}
};

class ExponentialFeature : public Feature
{
public:
    const int i;
    const double scale;
    ExponentialFeature(int i, double scale) : i(i), scale(scale) {}
    double get_value(vec x) override {return exp(scale * x(i));}
    void get_name() override {cout << "exp(" << scale << " * " << "x" << i << ")";}
};

class Library
{
    // 
    private:

        const unordered_map<string, int> feature_map = {
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

        vector<Feature*> features; // vector of all features
        double m; // number of model features
        const double N; // number of variables + drivers. Assumes an enhanced state vector z = (x_0, x_1 ... x_n, u_0, u_1 ... u_s)

        // add all features based on input
        vector<Feature*> build_features(vector<string> feature_list, double N)
        {
            vector<Feature*> lib;
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

        void add_feature(vector<Feature*> &lib, Feature* new_feature)
        {
            lib.push_back(new_feature);
            m += 1;
        }

    public:

        // constructors
        Library(int N) : N(N) {} // empty library
        Library(vector<string> feature_list, int N) : 
            N(N), 
            features(build_features(feature_list, N)) 
            {}

        // add new feature to library
        void add_feature(Feature* new_feature)
        {
            features.push_back(new_feature);
            m += 1;
        }

        // get value from one feature
        double get_value(vec z, Feature* feature)
        {
            return feature -> get_value(z);
        }

        // get values, create new vector
        vec get_values(vec z)
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
        void get_values(vec &theta, vec z)
        {
            int i = 0;
            for(Feature* &feature : features)
            {
                theta(i) = get_value(z, feature);
                ++i;
            }
        }

        void get_name(Feature* feature)
        {
            feature -> get_name();
        }

        void get_names()
        {
            for (Feature* &feature : features)
            {
                get_name(feature);
            }
        }

};


// int main(int argc, char **argv)
// {

//     vec d(3);
//     d(0) = 1.0;
//     d(1) = 10.0;
//     d(2) = 1.0;


//     // std::vector<Feature*> lib;
//     vector<string> feature_list = {"poly_1", "poly_2", "poly_3"};
//     Library lib = Library(feature_list, 3);

//     lib.get_names();

//     vec test = lib.get_values(d);
//     for (auto &val : test)
//     {
//         cout << "The value is: " << val << endl;
//     }

//     return 0;
// }