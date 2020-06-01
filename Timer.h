//
// Created by sakex on 06/09/2019.
//

#ifndef TRADING_TIMER_H
#define TRADING_TIMER_H

#include <chrono>
#include <string>
#include <iostream>

/// Namespace for util functions
namespace utils {

    /// Class used to time the different functions during training
    class Timer {
    public:
        /**
         * Constructor, starts timing right away
         * @param title String that contains the name of the function being timed
         */
        explicit Timer(std::string title);

        /// Prints the time elapsed
        void stop() const;

    private:
        using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
        std::string name;
        time_point start;
    };
}

#endif //TRADING_TIMER_H
