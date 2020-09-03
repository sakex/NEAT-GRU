/*
 * Generation.cpp
 *
 *  Created on: Sep 1, 2019
 *      Author: sakex
 */

#include "Generation.h"

namespace NeuralNetwork {

long Generation::_counter = 0;
std::unordered_map<Gene::coordinate, long> Generation::evolutions = {};
std::mutex Generation::mutex;

void Generation::reset(){
	evolutions.clear();
}

long Generation::number(Gene::coordinate const & coordinate){
	mutex.lock();
	if(evolutions.find(coordinate) == evolutions.end()){
		long new_value = ++_counter;
		evolutions[coordinate] = new_value;
		mutex.unlock();
		return new_value;
	}
	else{
		mutex.unlock();
		return evolutions[coordinate];
	}
}

} /* namespace NeuralNetwork */
