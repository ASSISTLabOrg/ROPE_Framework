#ifndef ROPE_VECTORSPACES_H   
#define ROPE_VECTORSPACES_H

namespace vectorSpaces{

struct Space
{
    void get_name();

};

struct FixedHeightGrid : public Space
{

};

struct UnstructuredGrid : public Space
{

};

struct LatentSpace : public Space
{

};

struct EigenSpace : public Space
{
    matrix U; // (n,r) matrix

    double interpolate();
};

};

#endif