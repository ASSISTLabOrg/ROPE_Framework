

class PCA
{

    const matrix U;


    double inner_product_operator(const vec& a, const vec& z)
    {   
        double x = 0.0;
        for (int i=0 ; i<a.size() ; ++i)
        {
            for (int j=0 ; i<z.size() ; ++i)
            {
                x += a(i) * U(i, j) * z(j);
            }
        }
        return x;
    }

    static double euclidean_distance(const vec& x, const vec& y)
    {
        double val = 0.0;
        for (int i=0 ; i<x.size() ; ++i)
        {
            val += std::cmath::pow(x(i) - y(i), 2);
        }
        return std::cmath::sqrt(val);
    }

    double function_of_point(const double& pt)
    {
        return get_position_from_point(pt)
    }

    state_vector get_nearest(const vec& x, const double& n)
    {
        state_vector a(n);
        tree_function(&a, x, n);
        return a;
    }
    
    state_vector get_weights(const vec& pts, const vec& x)
    {
        std::vector weights;
        for(const double& pt : pts)
        {
            weights.push_back(1 / euclidean_distance(x, function_of_point(pt)));
        }
        return weights;
    }

    double interpolate(const double& lat, const double& lon, const double& alt, const double& n, const double& z)
    {
        state_vector pts = get_nearest(lat, lon, alt, n);
        state_vector wghts = get_weights(pts, x);
        
        return inner_product_operator(wghts, z);
    }

}