#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <numeric> 
#include <functional>
#include <map>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

struct Variable
{
    virtual void put(const double& /* t */, const double& /*value*/) {}
    virtual double get(const double& /* t */) {return 0.0;}
};

struct StateVariable : public Variable
{
    std::vector<double> _t;
    std::vector<double> _values;

    StateVariable() {_t.push_back(0); _values.push_back(0);}
    StateVariable(const double& t_0, const double& v_0) {_t.push_back(t_0); _values.push_back(v_0);}
    
    void put(const double& t, const double& value)
    {
        _t.push_back(t);
        _values.push_back(value);
    }

    double get(const double& t){
        auto it = std::lower_bound(_t.begin(), _t.end(), t);
        if (it == _t.begin()) {
            return *_values.begin();
        } else if (it == _t.end()) {
            return *_values.rbegin();
        } else {
            auto i = std::distance(_t.begin(), it - 1);
            auto j = std::distance(_t.begin(), it);
            double v = _values[i] + (_values[j] - _values[i]) / (_t[j] - _t[i]) * (t - _t[i]);
            return v;
        }
    }

    double get(const int& i) {
        return _values[i];
    }
};

struct ControlVariable : public Variable
{
    const std::vector<double> _t;
    const std::vector<double> _values;
    ControlVariable(std::vector<double> t, std::vector<double> values) : _t(t), _values(values) {}

    void put(const double& /* t */, const double& /* value */) {}

    double get(const double& t){
        auto it = std::lower_bound(_t.begin(), _t.end(), t);
        if (it == _t.begin()) {
            return *_values.begin();
        } else if (it == _t.end()) {
            return *_values.rbegin();
        } else {
            auto i = std::distance(_t.begin(), it - 1);
            auto j = std::distance(_t.begin(), it);
            double x = _values[i] + (_values[j] - _values[i]) / (_t[j] - _t[i]) * (t - _t[i]);
            return x;
        }
    }
};

struct State
{
    std::vector<Variable*> _vars;

    State(std::vector<Variable*> vars) : _vars(vars) {}

    std::vector<double> get(const double& t)
    {
        std::vector<double> result;
        for (Variable* &var : _vars)
        {
            result.push_back(var -> get(t));
        }
        return result;
    }

    std::vector<double> get(const int& i)
    {
        std::vector<double> result;
        for (Variable* &var : _vars)
        {
            result.push_back(var -> get(i));
        }
        return result;
    }

    void put(const double& t, const double& value)
    {
        std::vector<double> result;
        for (Variable* &var : _vars)
        {
            var -> put(t, value);
        }
    }
};

// double Library::get_value(const double&t, const state_vector& z, const driver_vector& u, Feature* feature);
// {
//     return feature -> get_value(t, z, u);
// }


int main(int argc, char **argv){

    double N = 100;
    double dt = 0.01;
    std::vector<double> t;
    std::vector<double> u;
    for (double i=0 ; i < N ; ++i) {
        t.push_back(i * dt);
        u.push_back(std::cos(2.0 * 3.14 * i * dt));
    }
    std::vector<Variable*> vars;
    vars.push_back(new StateVariable());
    vars.push_back(new ControlVariable(t, u));
    State state = State(vars);

    for (double i=0 ; i < N - 1 ; ++i) {
        double diff = std::cos(2.0 * 3.14 * (i * dt + dt / 2.0)) - state.get(i * dt + dt / 2.0)[1];
        std::cout << i * dt + dt / 2.0 << "\t" << diff << std::endl;
    }

    // std::cout << driver.get(7.15) << "\t" << std::cos(2.0 * 3.14 * 7.15) << std::endl;

    return 0;
}