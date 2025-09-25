

class PCA
{

    const mat U;


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

    vec get_nearest(const vec& x, const double& n)
    {
        vec a(n);
        return a;
    }
    
    vec get_weights(const vec& pts, const vec& x)
    {
        vec wghts(pts.size());
        return weights;
    }

    double interpolate(const double& lat, const double& lon, const double& alt, const double& n)
    {
        vec pts = get_nearest(lat, lon, alt, n);
        vec wghts = get_weights(pts, x);
    }

}