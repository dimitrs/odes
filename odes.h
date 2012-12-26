#ifndef ODES
#define ODES

template <class InputIterator, class OutputIterator, class T, class Time=double>
std::pair<std::vector<Time>, OutputIterator> feuler(InputIterator inBegin, InputIterator inEnd, Time x0, Time x1, int nr_steps, T f, OutputIterator yout)
{
    typedef typename std::iterator_traits<InputIterator>::iterator_category Category;
    return __feuler(inBegin, inEnd, x0, x1, nr_steps, f, yout, Category());
}


template <class InputIterator, class OutputIterator, class T, class Time=double>
std::pair<std::vector<Time>, OutputIterator> runge_kutta4(InputIterator inBegin, InputIterator inEnd, Time x0, Time x1, int nr_steps, T f, OutputIterator yout)
{
    typedef typename std::iterator_traits<InputIterator>::iterator_category Category;
    return __runge_kutta4(inBegin, inEnd, x0, x1, nr_steps, f, yout, Category());
}




template <class InputIterator, class RandomAccessIterator, class T, class Time=double>
std::pair<std::vector<Time>, RandomAccessIterator> __feuler(InputIterator inBegin, InputIterator inEnd, Time x0, Time x1, int nr_steps, T f, RandomAccessIterator yout, std::random_access_iterator_tag)
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    auto delta_t = static_cast<Time>((x1-x0)/nr_steps);
    auto nr_equations = inEnd - inBegin;

    std::copy(inBegin, inEnd, yout);
    std::vector<Time> ytime(nr_steps);

    for (auto ii=0, i=0; ii< nr_steps; ++ii) {
        std::vector<value_type> y(nr_equations);
        f(yout+i, x0, begin(y));

        for (int u = 0; u<nr_equations; ++u) {
            yout[2+i+u] = yout[i+u] + y[u]*delta_t;
        }
        i += 2;

        ytime[ii] = x0;
        x0 += delta_t;
    }
    yout += (nr_steps*nr_equations)+nr_equations;

    return std::pair<std::vector<Time>, RandomAccessIterator>(ytime, yout);
}

template <class InputIterator, class BidirectionalIterator, class T, class Time=double>
std::pair<std::vector<Time>, BidirectionalIterator> __feuler(InputIterator inBegin, InputIterator inEnd, Time x0, Time x1, int nr_steps, T f, BidirectionalIterator yout, std::bidirectional_iterator_tag)
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    auto delta_t = static_cast<Time>((x1-x0)/nr_steps);
    auto nr_equations = 0;
    BidirectionalIterator it = yout;
    std::for_each(inBegin, inEnd, [&nr_equations, &yout](value_type& v){ *yout++ = v; nr_equations++; } );
    std::vector<Time> ytime(nr_steps);

    for (auto ii=0; ii< nr_steps; ++ii) {
        std::vector<value_type> y(nr_equations);
        typename std::vector<value_type>::iterator res = f(it, x0, begin(y));
        for (auto i=0; i<nr_equations; ++i) {
            *yout++ = *it++ + (*res++)*delta_t;
        }
        ytime.push_back(x0);
        x0 += delta_t;
    }

    return std::pair<std::vector<Time>, BidirectionalIterator>(ytime, yout);
}



template <class InputIterator, class RandomAccessIterator, class T, class Time=double>
std::pair<std::vector<Time>, RandomAccessIterator> __runge_kutta4(InputIterator inBegin, InputIterator inEnd, Time x0, Time x1, int nr_steps, T f, RandomAccessIterator yout, std::random_access_iterator_tag)
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;
    auto delta_t = static_cast<Time>((x1-x0)/nr_steps);
    auto nr_equations = inEnd - inBegin;
    yout = std::copy(inBegin, inEnd, yout);

    std::vector<Time> ytime(nr_steps);
    for (auto ii=0; ii< nr_steps; ++ii) {
        typedef typename std::vector<value_type> vec;

        // K1 = f(x, u(x))
        vec v1(nr_equations);
        typename vec::iterator k1 = f(yout-nr_equations, x0, begin(v1));

        // K2 = f(x+deltaX/2, u(x)+K1*deltaX/2)
        vec v2(nr_equations);
        vec y2(nr_equations);
        std::transform(yout-nr_equations, yout, k1, begin(y2), [delta_t](value_type& y, value_type& k) {
            return y+0.5*delta_t*k; });
        typename vec::iterator k2 = f(begin(y2), 0.5*delta_t+x0, begin(v2));

        // K3 = f(x+deltaX/2, u(x)+K2*deltaX/2)
        vec v3(nr_equations);
        vec y3(nr_equations);
        std::transform(yout-nr_equations, yout, k2, begin(y3), [delta_t](value_type& y, value_type& k){
            return y+0.5*delta_t*k; });
        typename vec::iterator k3 = f(begin(y3), 0.5*delta_t+x0, begin(v3));

        // K4 = f(x+deltaX, u(x)+K2*deltaX)
        vec v4(nr_equations);
        vec y4(nr_equations);
        std::transform(yout-nr_equations, yout, k3, begin(y4), [delta_t](value_type& y, value_type& k){
            return y+delta_t*k; });
        typename vec::const_iterator k4 = f(begin(y4), delta_t+x0, begin(v4));

        // u(x+deltaX) = u(x) + (1/6) (K1 + 2 K2 + 2 K3 + K4) * deltaX
        for (auto i=0; i<nr_equations; ++i,++yout) {
            *yout = *(yout-nr_equations) + (1.0/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) * delta_t;
        }
        ytime[ii] = x0;
        x0 += delta_t;
    }

    return std::pair<std::vector<Time>, RandomAccessIterator>(ytime, yout);
}


#endif