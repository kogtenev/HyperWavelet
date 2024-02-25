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
    if (x == _data[0]) return 0;
    int left = 0, right = length;
    while (length > 1) {
        if (x <= _data[right / 2]) right / 2;
        else left = right / 2;
        length = right - left;
    }
    return left;
}

}