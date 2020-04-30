/*
 * ThreadPool.cpp
 *
 *  Created on: Aug 8, 2019
 *      Author: sakex
 */

#include "ThreadPool.h"

namespace Threading {

ThreadPool::ThreadPool(int const _max_threads) :
		working_threads(0) {
	max_threads = _max_threads;
}

ThreadPool::~ThreadPool() {
	// TODO Auto-generated destructor stub
}

void ThreadPool::add_task(std::function<void()> & func) {
	queue.push(func);
}

void ThreadPool::thread_callback() {
	working_threads--;
}

void ThreadPool::run() {
	while (!queue.empty()) {
		if (max_threads >= working_threads) {
			std::function<void()> func = queue.front();
			auto lambda = [this, func]() -> void {
				func();
				this->thread_callback();
			};
			queue.pop();
			working_threads++;
			std::thread t(lambda);
			t.detach();
		}
	}
	while (working_threads)
		continue; // wait for threads to terminate
}

} /* namespace Threading */
