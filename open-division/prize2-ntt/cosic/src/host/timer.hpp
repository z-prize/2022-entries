#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>

class Timer
{
    std::chrono::high_resolution_clock::time_point mTimeStart;

public:
    Timer() { reset(); }
    long long stop()
    {
        std::chrono::high_resolution_clock::time_point timeEnd = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - mTimeStart).count();
    }
    void reset() { mTimeStart = std::chrono::high_resolution_clock::now(); }
};

// helper macro to time a specific operation
#define TIME(op)             \
    ;                        \
    timer.reset();           \
    op;                      \
    stopTime = timer.stop(); \
    std::cout << "(timing: " << stopTime << " microseconds)" << std::endl;

// helper macro to time a for loop
#define TIMEFOR(forl)            \
    ;                            \
    timer.reset();               \
    forl                         \
        stopTime = timer.stop(); \
    std::cout << "(timing: " << stopTime << " microseconds)" << std::endl;

#endif