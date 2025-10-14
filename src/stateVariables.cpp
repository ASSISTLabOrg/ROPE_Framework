#include stateVariables.h

namespace stateVariables{

    virtual void Variable::put(const double& /* t */, const double& /*value*/) {}

    virtual double Variable::get(const double& /* t */) { return 0.0; }

    virtual double Variable::get(const double& /* i */) { return 0.0; }

    StateVariable::StateVariable() { put(0.0, 0.0); }

    StateVariable::StateVariable(const double& t_0, const double& v_0) { put(t_0, v_0); }

    void StateVariable::put(const double& t, const double& value) {
        _t.push_back(t);
        _values.push_back(value);
    }

    double StateVariable::get(const double& t) {
        auto it = std::lower_bound(_t.begin(), _t.end(), t);
        if (it == _t.begin()) {
            return *_values.begin();
        } 
        else if (it == _t.end()) {
            return *_values.rbegin();
        } 
        else {
            auto i = std::distance(_t.begin(), it - 1);
            auto j = std::distance(_t.begin(), it);
            double v = _values[i] + (_values[j] - _values[i]) / (_t[j] - _t[i]) * (t - _t[i]);
            return v;
        }
    }

    double StateVariable::get(const int& i) {
        return _values[i];
    }  

    ControlVariable(std::vector<double> t, std::vector<double> values);
    void put(const double& /* t */, const double& /* value */);
    double get(const double& t);
    double get(const int& i);

    {
        std::vector<Variable*> _vars;
        State(std::vector<Variable*> vars);
        std::vector<double> get(const double& t);
        std::vector<double> get(const int& i);
        void put(const double& t, const double& value);
    };

};