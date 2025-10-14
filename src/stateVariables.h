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

    struct Variable {
        virtual void put(const double& /* t */, const double& /*value*/);
        virtual double get(const double& /* t */);
    };

    struct StateVariable : public Variable {
        std::vector<double> _t;
        std::vector<double> _values;
        StateVariable();
        StateVariable(const double& t_0, const double& v_0);
        void put(const double& t, const double& value);
        double get(const double& t);
        double get(const int& i);
    };

    struct ControlVariable : public Variable
    {
        const std::vector<double> _t;
        const std::vector<double> _values;
        ControlVariable(std::vector<double> t, std::vector<double> values);
        void put(const double& /* t */, const double& /* value */);
        double get(const double& t);
        double get(const int& i);
    };

    struct State
    {
        std::vector<Variable*> _vars;
        State(std::vector<Variable*> vars);
        std::vector<double> get(const double& t);
        std::vector<double> get(const int& i);
        void put(const double& t, const double& value);
    };

};
#endif