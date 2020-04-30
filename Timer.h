//
// Created by sakex on 06/09/2019.
//

#ifndef TRADING_TIMER_H
#define TRADING_TIMER_H

#include <chrono>
#include <string>
#include <iostream>

namespace utils {
    class Timer {
    public:
        explicit Timer(std::string);

        void stop() const;

    private:
        using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
        std::string name;
        time_point start;
    };
}

#endif //TRADING_TIMER_H
