#include <string>
#include <iostream>
#include <vector>
#include <sstream>

std::vector<std::string> split(std::string& s, const std::string& delimiter)
{
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);
    return tokens;
}

int main(int argc, char **argv){

    std::vector<std::string> teststrs = {};

    for (int i=0 ; i < 3 ; ++i)
    {
        std::stringstream ss;
        ss << "poly:x_" << i;
        teststrs.push_back(ss.str());
        ss.str(std::string());
    }

    std::vector<double> x = {1, 10, 17};

    for (int i=0 ; i < 3 ; ++i)
    {
        for (int j=i ; j < 3 ; ++j)
        {
            std::stringstream ss;
            ss << "poly:x_" << i << "*x_" << j;
            teststrs.push_back(ss.str());
            ss.str(std::string());
        }
    }

    for (std::string& str : teststrs)
    {
        std::vector<std::string> tokens = split(str, ":");
        for (int i=0 ; i<tokens.size() ; ++i)
        {
            if (i == 0)
            {
                std::cout << tokens[i] << "\t";
            }

            else
            {
                std::vector<std::string> vars 
                std::vector<std::string> vars = split(tokens[i], "*");
                for (std::string var : vars)
                {
                    std::vector<std::string> var_num = split(var, "_");
                    int n = std::stoi(var_num[1]);
                    std::cout << "Variable: " << var_num[0] << ", Index: " << n << ", Value: " << x[n] << "\t";
                }
            }
        }
        std::cout << std::endl;
    }

    return 0;
}