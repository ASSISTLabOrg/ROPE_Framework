#include "stateVariables.h"

namespace stateVariables{

    void Variable::put(const double& /* t */, const double& /*value*/) {}

    double Variable::get_value(const double& /* t */) { return 0.0; }

    double Variable::get_value(const int& /* i */) { return 0.0; }

    StateVariable::StateVariable() { put(0.0, 0.0); }

    StateVariable::StateVariable(const double& t_0, const double& v_0) { put(t_0, v_0); }

    void StateVariable::put(const double& t, const double& value) {
        _t.push_back(t);
        _values.push_back(value);
    }

    double StateVariable::get(const double& t) {
        return linterp(t, _t, _values);
    }

    double StateVariable::get(const int& i) {
        return _values[i];
    }

    ControlVariable::ControlVariable(std::vector<double> t, std::vector<double> values) : _t(t), _values(values) {}

    void ControlVariable::put(const double& /* t */, const double& /* value */) {}

    double ControlVariable::get(const double& t)
    {
        return linterp(t, _t, _values);
    }

    double ControlVariable::get(const int& i)
    {
        return _values[i];
    }

    State::State(std::vector<Variable*> vars) : _vars(vars) {}
    
    void State::put(const std::vector<double>& z)
    {
        for (int i=0 ; i<z.size() ; ++i)
        {
            _vars[i] -> put(z[i]);
        }
    }

    std::vector<double> State::get(const double& t)
    {
        std::vector<double> _q;
        for (int i=0 ; i<z.size() ; ++i)
        {
            _q.push_back(_vars[i] -> get(t));
        }
    }

    std::vector<double> State::get(const int& i)
    {
        std::vector<double> _q;
        for (int j=0 ; j<z.size() ; ++j)
        {
            _q.push_back(_vars[j] -> get(i));
        }
    }

};