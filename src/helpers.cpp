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

}