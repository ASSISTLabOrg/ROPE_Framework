#include "../sindy/features.h"

using namespace std;
using namespace features;

int main(int argc, char **argv)
{

    vec d(3);
    d(0) = 1.0;
    d(1) = 10.0;
    d(2) = 1.0;

    vector<string> feature_list = {"poly_1", "poly_2", "poly_3"};
    Library lib = Library(feature_list, 3);

    lib.get_names();

    vec test = lib.get_values(d);
    for (auto &val : test)
    {
        cout << "The value is: " << val << endl;
    }

    return 0;
}