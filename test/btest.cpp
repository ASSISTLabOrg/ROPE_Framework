
#include <iostream>
#include <fstream>
#include <string>
#include <numeric> 
#include <boost/variant.hpp>

struct Point{
    double a;
    void echo() {std::cout << a << std::endl;}
};

struct AnotherPoint{
    double a;
    double b;
    void echo() {std::cout << a << '\t' << b << std::endl;}
};

using MyVariant = boost::variant<Point, AnotherPoint>;

int main(int argc, char **argv){

    std::vector<MyVariant> x;

    Point p = Point();
    p.a = 7;
    AnotherPoint p1 = AnotherPoint();
    p1.a = 8;
    p1.b = 9;

    p.echo();
    p1.echo();

    x.push_back(p);
    // x.push_back(p1);

    x[0].echo();

    // x[0].echo();
    // x[1].echo();

    return 0;
}