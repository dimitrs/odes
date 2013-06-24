#include "odes.h"

#include <vector>
#include <chrono>
#include <blitz/array.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include <iterator>


float expected_results[11][2] = {
    { 0, 1 },
    { 0.1, 0.9 },
    { 0.161, 0.8119 },
    { 0.194718, 0.733626 },
    { 0.209595, 0.663589 },
    { 0.211711, 0.600576 },
    { 0.205438, 0.543655 },
    { 0.193907, 0.4921 },
    { 0.179341, 0.445329 },
    { 0.163305, 0.402865 },
    { 0.146874, 0.364302 } };

float expected_results_rk4[11][2] = {
    { 0, 1 },
    { 0.0818671, 0.904842 },
    { 0.134055, 0.818738 },
    { 0.164634, 0.740827 },
    { 0.179722, 0.670329 },
    { 0.183931, 0.606539 },
    { 0.180709, 0.548819 },
    { 0.172612, 0.496592 },
    { 0.161512, 0.449335 },
    { 0.148765, 0.406575 },
    { 0.135332, 0.367884 } };

// Given differential eqn.:   y'' + 3xy' + 7y =  cos(2x)
struct func_rhs {
    template <class InputIterator, class OutputIterator>
    OutputIterator operator()(InputIterator it, const double& t, OutputIterator yout) {
        typedef typename std::iterator_traits<InputIterator>::value_type value_type;
        value_type y1 = *it++;
        value_type y2 = *it++;
        yout[0] = pow(y2,2)-(2*y1);
        yout[1] = y1-y2-t*pow(y2,2);
        return yout;
    }
};

template <class Vector>
struct func_blitz {
    Vector operator()(const Vector& yin, const double t) {
        Vector y(2);
        y(0) = pow(yin(1),2)-(2*yin(0));
        y(1) = yin(0)-yin(1)-t*pow(yin(1),2);
        return y;
    }
};

template <class T>
std::pair<std::vector<double>, std::vector<double>::iterator> feuler_hand_written(std::vector<double>& out, const std::vector<double>& init, double x0, double x1, int nr_steps, T f)
{
    auto delta_t = static_cast<double>((x1-x0)/nr_steps);
    auto nr_equations = static_cast<int>(init.size());

    std::vector<double>::iterator yout = begin(out);
    std::copy(begin(init), end(init), yout);
    std::vector<double> ytime(nr_steps);

    for (auto ii=0, i=0; ii< nr_steps; ++ii) {
        std::vector<double> y(nr_equations);
        f(yout+i, x0, begin(y));
        out[2+i] = out[i] + y[0]*delta_t;
        out[3+i] = out[i+1] + y[1]*delta_t;
        i += 2;
        ytime[ii] = x0;
        x0 += delta_t;
    }
    yout += (nr_steps*nr_equations)+nr_equations;
    return std::pair<std::vector<double>, std::vector<double>::iterator>(ytime, begin(out));
}

template <class T, class Matrix, class Vector>
std::vector<double> hand_writen_rk4(Matrix& out, const Vector& init, double x0, double x1, int nr_steps, T f)
{
    std::vector<double> ytime(nr_steps);
    auto delta_t = static_cast<double>((x1-x0)/nr_steps);
    Vector u = init;

    for (auto ii=0; ii< nr_steps; ++ii) {
        // K1 = f(x, u(x))
        Vector k1 = f(u, x0);

        // K2 = f(x+deltaX/2, u(x)+K1*deltaX/2)
        Vector k2 = f(u+0.5*delta_t*k1, 0.5*delta_t+x0);

        // K3 = f(x+deltaX/2, u(x)+K2*deltaX/2)
        Vector k3 = f(u+0.5*delta_t*k2, 0.5*delta_t+x0);

        // K4 = f(x+deltaX, u(x)+K2*deltaX)
        Vector k4 = f(u+delta_t*k3, delta_t+x0);

        // u(x+deltaX) = u(x) + (1/6) (K1 + 2 K2 + 2 K3 + K4) * deltaX
        u = u + (1.0/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4) * delta_t;


        boost::numeric::ublas::matrix_row<Matrix> mr(out, ii);
        mr = u;

        ytime[ii]=x0;
        x0 += delta_t;
    }
    return ytime;
}

int main()
{
    double tmin = 0.0;
    double tmax = 1.0;
    double delta_t = 0.1;
    int nr_steps = int((tmax-tmin)/delta_t);
    auto nr_equations = 2;

    { // std::vector runge_kutta4 unit test
        typedef std::vector<double> Vector;

        Vector out(nr_steps*nr_equations+nr_equations);
        Vector init = {0, 1};

        std::pair<std::vector<double>, Vector::iterator >  o =
            odes::runge_kutta4(init.begin(), init.end(), tmin, tmax, nr_steps, func_rhs(), out.begin());

        Vector::iterator it = out.begin();
        Vector::iterator itend = o.second;
        int i = 0;
        for (; it != itend; ++it) {
            if (fabs((*it)-expected_results_rk4[i][0]) > 0.00001) break;
            ++it;
            if (fabs((*it)-expected_results_rk4[i][1]) > 0.00001) break;
            i++;
        }
        if (i==nr_steps+1)  std::cout << "PASS: runge_kutta4 std::vector \n";
        else std::cout << "FAIL: runge_kutta4 std::vector \n";
    }

    {  // UBLAS runge_kutta4 unit test
        typedef boost::numeric::ublas::matrix<double> Matrix;

        Matrix init(1,nr_equations);
        init(0,0) = 0;
        init(0,1) = 1;

        Matrix out(nr_steps+1, nr_equations);
        std::pair<std::vector<double>, Matrix::iterator2 >  o =
            odes::runge_kutta4(init.begin2(), init.end2(), tmin, tmax, nr_steps, func_rhs(), out.begin2());

        unsigned i = 0;
        for (; i < out.size1(); ++i) {
            if (fabs((out(i,0))-expected_results_rk4[i][0]) > 0.00001) break;
        }
        if (i==unsigned(nr_steps+1)) std::cout << "PASS: runge_kutta4 UBLAS\n";
        else std::cout << "FAIL: runge_kutta4 UBLAS\n";
    }

    { // C array runge_kutta4 unit test
        double out[nr_steps*nr_equations];
        double init[] = {0, 1};
        std::pair<std::vector<double>, double* >  o =
            odes::runge_kutta4(&init[0], &init[nr_equations], tmin, tmax, nr_steps, func_rhs(), out);
        int j=0;
        for (int i=0; i < nr_steps*nr_equations; i += nr_equations, j++) {
            if (fabs((out[i])-expected_results_rk4[j][0]) > 0.00001) {
                break;
            }
        }
        if (j==nr_steps)
            std::cout << "PASS: runge_kutta4 C-array\n";
        else
            std::cout << "FAIL: runge_kutta4 C-array\n";
    }

    { // Blitz runge_kutta4 unit test
        typedef blitz::Array<double,2> Vector;
        Vector init(1,nr_equations);
        init = 0, 1;
        Vector out(nr_steps+nr_equations, nr_equations);

        std::pair<std::vector<double>, Vector::iterator >  o =
            odes::runge_kutta4(init.begin(), init.end(), tmin, tmax, nr_steps, func_rhs(), out.begin());

        Vector::iterator it = out.begin();
        Vector::iterator itend = o.second;
        int i = 0;
        for (; it != itend; ++it) {
            if (fabs((*it)-expected_results_rk4[i][0]) > 0.00001) {
                break;
            }
            ++it;
            if (fabs((*it)-expected_results_rk4[i][1]) > 0.00001) {
                break;
            }
            i++;
        }
        if (i==nr_steps+1)
            std::cout << "PASS: runge_kutta4 Blitz \n";
        else
            std::cout << "FAIL: runge_kutta4 Blitz \n";
    }

    { // std::vector feuler unit test
        typedef std::vector<double> Vector;

        Vector out(nr_steps*nr_equations+nr_equations);
        Vector init = {0, 1};
        std::pair<std::vector<double>, Vector::iterator >  o =
            odes::feuler(init.begin(), init.end(), tmin, tmax, nr_steps, func_rhs(), out.begin());

        Vector::iterator it = out.begin();
        Vector::iterator itend = o.second;
        int i = 0;
        for (; it != itend; ++it) {
            if (fabs((*it)-expected_results[i][0]) > 0.00001) {
                break;
            }
            ++it;
            if (fabs((*it)-expected_results[i][1]) > 0.00001) {
                break;
            }
            i++;
        }
        if (i==nr_steps+1)
            std::cout << "PASS: feuler std::vector \n";
        else
            std::cout << "FAIL: feuler std::vector \n";
    }

    { // Blitz feuler unit test
        typedef blitz::Array<double,2> Vector;
        Vector init(1,nr_equations);
        init = 0, 1;
        Vector out(nr_steps+nr_equations, nr_equations);

        std::pair<std::vector<double>, Vector::iterator >  o =
            odes::feuler(init.begin(), init.end(), tmin, tmax, nr_steps, func_rhs(), out.begin());

        Vector::iterator it = out.begin();
        Vector::iterator itend = o.second;
        int i = 0;
        for (; it != itend; ++it) {
            if (fabs((*it)-expected_results[i][0]) > 0.00001) {
                break;
            }
            ++it;
            if (fabs((*it)-expected_results[i][1]) > 0.00001) {
                break;
            }
            i++;
        }
        if (i==nr_steps+1)
            std::cout << "PASS: feuler Blitz \n";
        else
            std::cout << "FAIL: feuler Blitz \n";
    }

    { // C array feuler unit test
        double out[nr_steps*nr_equations];
        double init[] = {0, 1};
        std::pair<std::vector<double>, double* >  o =
            odes::feuler(&init[0], &init[nr_equations], tmin, tmax, nr_steps, func_rhs(), out);
        int j=0;
        for (int i=0; i < nr_steps*nr_equations; i += nr_equations, j++) {
            if (fabs((out[i])-expected_results[j][0]) > 0.00001) {
                break;
            }
        }
        if (j==nr_steps)
            std::cout << "PASS: feuler C-array\n";
        else
            std::cout << "FAIL: feuler C-array\n";
    }

    {  // UBLAS feuler unit test
        typedef boost::numeric::ublas::matrix<double> Matrix;

        Matrix init(1,nr_equations);
        init(0,0) = 0;
        init(0,1) = 1;

        Matrix out(nr_steps+1, nr_equations);
        std::pair<std::vector<double>, Matrix::iterator2 >  o =
            odes::feuler(init.begin2(), init.end2(), tmin, tmax, nr_steps, func_rhs(), out.begin2());

        unsigned i = 0;
        for (; i < out.size1(); ++i) {
            if (fabs((out(i,0))-expected_results[i][0]) > 0.00001) {
                break;
            }
        }
        if (i==unsigned(nr_steps+1))
            std::cout << "PASS: feuler UBLAS\n";
        else
            std::cout << "FAIL: feuler UBLAS\n";
    }

    { // std::vector feuler performance test
        typedef std::vector<double> Vector;

        Vector out(nr_steps*nr_equations+nr_equations);
        Vector init = {0, 1};
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 900000; ++i)
            odes::feuler(init.begin(), init.end(), tmin, tmax, nr_steps, func_rhs(), out.begin());
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "feuler std::vector        time: " << elapsed.count() << '\n';
    }

    { // hand-written feuler performance test
        typedef std::vector<double> Vector;
        Vector out(nr_steps*nr_equations+nr_equations);
        Vector init = {0, 1};
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 900000; ++i)
            feuler_hand_written(out, init, tmin, tmax, nr_steps, func_rhs());
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "feuler hand-written       time: " << elapsed.count() << '\n';
    }

    {  // runge_kutta4 hand-written performance test

        typedef boost::numeric::ublas::matrix<double> Matrix;
        typedef boost::numeric::ublas::vector<double> Vector;

        Vector init(nr_equations);
        init(0) = 0;
        init(1) = 1;

        Matrix out(nr_steps+1, nr_equations);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 500000; ++i)
            hand_writen_rk4(out, init, tmin, tmax, nr_steps, func_blitz<Vector>());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "runge_kutta4 handwritten-ublas time: " << elapsed.count() << '\n';
    }

    { // runge_kutta4 std::vector performance test
        typedef std::vector<double> Vector;
        Vector out(nr_steps*nr_equations+nr_equations);
        Vector init = {0, 1};
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 500000; ++i)
            odes::runge_kutta4(init.begin(), init.end(), tmin, tmax, nr_steps, func_rhs(), out.begin());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "runge_kutta4 std::vector       time: " << elapsed.count() << '\n';
    }

    { // Blitz runge_kutta4 performance test
        typedef blitz::Array<double,2> Vector;
        Vector init(1,nr_equations);
        init = 0, 1;
        Vector out(nr_steps+nr_equations, nr_equations);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 500000; ++i)
            odes::runge_kutta4(init.begin(), init.end(), tmin, tmax, nr_steps, func_rhs(), out.begin());
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "runge_kutta4 Blitz       time: " << elapsed.count() << '\n';
    }

    {  // UBLAS runge_kutta4 preformance test
        typedef boost::numeric::ublas::matrix<double> Matrix;

        Matrix init(1,nr_equations);
        init(0,0) = 0;
        init(0,1) = 1;

        Matrix out(nr_steps+1, nr_equations);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 500000; ++i)
            odes::runge_kutta4(init.begin2(), init.end2(), tmin, tmax, nr_steps, func_rhs(), out.begin2());
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "runge_kutta4 ublas             time: " << elapsed.count() << '\n';
    }

    return 0;
}




