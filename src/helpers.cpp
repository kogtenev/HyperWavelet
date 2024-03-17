#include "helpers.h"

namespace hyper_wavelet {

int IntPow(int a, int power) {
    int result = 1;
    while (power > 0) {
        if (power % 2 == 0) {
            power /= 2;
            a *= a;
        } else {
            power--;
            result *= a;
        }
    }
    return result;
}

Profiler::Profiler() {
    _time = omp_get_wtime();
}

void Profiler::Tic() {
    _time = omp_get_wtime();
}

double Profiler::Toc() {
    return omp_get_wtime() - _time;
}

SegmentTree::SegmentTree(
    const double a, const double b, int nSegments
): _data(Eigen::ArrayXd::LinSpaced(Eigen::Sequential, nSegments + 1, a, b)) 
{
    if (a >= b) {
        throw std::invalid_argument("SegmentTree: a must be smaller than b!");
    }
}

int SegmentTree::Find(double x) const {
    int length = _data.size() - 1;
    if (x < _data[0] || x > _data[length]) return -1;
    int left = 0, right = length;
    while (length > 1) {
        if (x <= _data[left + length / 2]) right = left + length / 2;
        else left = left + length / 2;
        length = right - left;
    }
    return left;
}

void CartesianToSphere(const Eigen::Vector3d& n, double& phi, double& theta) {
    if (n[2] >= 1.) {
        phi = 0.;
        theta = M_PI_2;
        return;
    } else if (n[2] <= -1.) {
        phi = 0.;
        theta = -M_PI_2;
        return;
    }
    theta = std::asin(n[2]);
    if (n[0] == 0. && n[1] >= 0.) {
        phi = M_PI_2;
        return;
    } else if (n[0] == 0. && n[1] < 0.) {
        phi = 3 * M_PI / 2;
        return;
    }
    phi = std::atan(n[1] / n[0]);
    if ((n[0] >= 0. && n[1] < 0.) || (n[0] < 0. && n[1] <= 0.)) {
        phi += M_PI;
    } else if (n[0] < 0. && n[1] > 0) {
        phi += 2 * M_PI;
    }
}

}