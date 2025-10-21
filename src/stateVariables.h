#ifndef ROPE_MODELS_STATEVARIABLES_H   
#define ROPE_MODELS_STATEVARIABLES_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <numeric> 
#include <functional>
#include <boost/numeric/ublas/vector.hpp>

namespace stateVariables{

    template<typename T>
    auto linterp(const T& xn, const std::vector<T>& x, const std::vector<T>& y)
    {
        auto it = std::lower_bound(x.begin(), x.end(), xn);
        if (it == x.begin()) {
            return *y.begin();
        } 
        else if (it == x.end()) {
            return *y.rbegin();
        } 
        else {
            auto i = std::distance(x.begin(), it - 1);
            auto j = std::distance(x.begin(), it);
            double yn = y[i] + (y[j] - y[i]) / (x[j] - x[i]) * (xn - x[i]);
            return yn;
        }
    }

    struct Variable {
        virtual void put(const double& /* t */, const double& /*value*/);
        virtual double get(const double& /* t */);
        virtual double get(const int& /* i */);
    };

    typedef std::vector<std::unique_ptr<Variable>> state_vector;

    struct StateVariable : Variable {
        std::vector<double> _t;
        std::vector<double> _values;
        StateVariable();
        StateVariable(const double& t_0, const double& v_0);
        void put(const double& t, const double& value);
        double get(const double& t);
        double get(const int& i);
    };

    struct ControlVariable : Variable
    {
        const std::vector<double> _t;
        const std::vector<double> _values;
        ControlVariable(std::vector<double> t, std::vector<double> values);
        void put(const double& /* t */, const double& /* value */);
        double get(const double& t);
        double get(const int& i);
    };

    struct State {

        state_vector _vars;
        State(state_vector vars);
        void put(const std::vector<double>& z);
        std::vector<double> get(const double& t);
        std::vector<double> get(const int& i);

    };

};
#endif