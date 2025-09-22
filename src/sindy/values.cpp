
class Value
{
    string label;

};

class Variable : public Value
{
    string label;
    vec values;

    update(x)
    {
        values.push_back(x);
    }
    
};

class Driver : public Value
{

};

class State
{
    vector<Value*> values;
    vector<vector<Double>> state;

    update_state()
    {
        for (auto &var : variables)
            var -> update(x[i]);
    }
}