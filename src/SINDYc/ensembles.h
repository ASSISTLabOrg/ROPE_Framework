#ifndef ROPE_MODELENSEMBLES_H   
#define ROPE_MODELENSEMBLES_H

#include <iostream>
#include <fstream>
#include <string>
#include <numeric> 
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include "neutralEstimators.h"

namespace modelEnsembles{

using neutralEstimators::NeutralEstimator;

typedef boost::numeric::ublas::vector<double> state_vector;
typedef boost::numeric::ublas::matrix<double> matrix;
typedef std::vector<NeutralEstimator*> ensemble_vector;

class Ensemble {

    const ensemble_vector ensemble;

    Ensemble(std::string config_file);
    Ensemble(ensemble_vector models);

    void get_names();
    void predict(auto& inputs);

};

};
#endif