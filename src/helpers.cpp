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

double GetTime() {
    std::timespec ts;
    timespec_get(&ts, TIME_UTC);
    return 1. * ts.tv_sec + 1e-9 * ts.tv_nsec;
}

Profiler::Profiler() {
    _time = GetTime();
}

void Profiler::Tic() {
    _time = GetTime();
}

double Profiler::Toc() {
    return GetTime() - _time;
}

}