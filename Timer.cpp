//
// Created by sakex on 06/09/2019.
//

#include "Timer.h"

namespace utils {
    Timer::Timer(std::string title) :
            name(std::move(title)),
            start(std::chrono::high_resolution_clock::now()) {
    }

    void Timer::stop() const {
        time_point stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                stop - start);
        std::cout << "TIME ELAPSED <" + name + ">: " << duration.count() << std::endl;
    }
}