#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <boost/numeric/ublas/vector.hpp>
#include <numeric> 
#include <functional>

using namespace std;
typedef boost::numeric::ublas::vector< double > vec;

template<typename Type>
auto product(const std::vector<Type>& vec, Type init)
{
    return std::accumulate(vec.cbegin(), vec.cend(), init, std::multiplies<Type>{});
}

class Feature{
public:
    virtual double get_value(vec x) {return 1.0;};
};

class LinearFeature : public Feature
{
public:
    const int i;
    LinearFeature(int i) : i(i) {}
    double get_value(vec x) override {return x(i);}
};

class QuadraticFeature : public Feature
{
public:
    const int i;
    const int j;
    QuadraticFeature(int i, int j) : i(i), j(j) {}
    double get_value(vec x) override {return x(i) * x(j);}
};

class CubicFeature : public Feature
{
public:
    const int i;
    const int j;
    const int k;
    CubicFeature(int i, int j, int k) : i(i), j(j), k(k) {}
    double get_value(vec x) override {return x(i) * x(j) * x(k);}
};

class HighOrderPolynomialFeature : public Feature
{
public:
    const vector<int> idx;
    vector<double> _x;
    HighOrderPolynomialFeature(vector<int> idx) : idx(idx), _x(init_subvec(idx)) {}

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

    double get_value(vec x) override 
    {
        get_components(x);
        return product(_x, 1.0);
    }
};

class SinusoidalFeature : public Feature
{
public:
    const int i;
    const double scale;
    const double phase;
    SinusoidalFeature(int i, double scale, double phase) : i(i), scale(scale), phase(phase) {}
    double get_value(vec x) override {return sin(scale * x(i) + phase);}
};

class ExponentialFeature : public Feature
{
public:
    const int i;
    const double scale;
    ExponentialFeature(int i, double scale) : i(i), scale(scale) {}
    double get_value(vec x) override {return exp(scale * x(i));}
};

class FeatureFactory
{
public:
    Feature* makeFeature(int i) {return new LinearFeature(i);}
    Feature* makeFeature(int i, int j) {return new QuadraticFeature(i,j);}
    Feature* makeFeature(int i, int j, int k) {return new CubicFeature(i,j,k);}
    Feature* makeFeature(vector<int> idx) {return new HighOrderPolynomialFeature(idx);}
    Feature* makeFeature(int i, double scale, double phase = 0.0) {return new SinusoidalFeature(i,scale,phase);}
    Feature* makeFeature(int i, double scale) {return new ExponentialFeature(i,scale);}
};

class ModelBuilder
{
public:
    
};


int main(int argc, char **argv)
{

    vec d(3);
    d(0) = 1.0;
    d(1) = 10.0;
    d(2) = 1.0;


    std::vector<Feature*> lib;

    FeatureFactory factory;
    for (int i=0 ; i<3 ; ++i)
    {
        //lib.push_back(new PolynomialFeature(i)); 
        lib.push_back(factory.makeFeature(i));
    }


    vec test(3);
    for (int i=0; i<3 ; ++i)
    {
        test(i) = lib[i] -> get_value(d);
    }

    cout << test(0) << '\t' << test(1) << '\t' << test(2) << endl;

    return 0;
}