project(neat_gru)
cmake_minimum_required(VERSION 3.0)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -ffast-math -frename-registers -flto")
set(CMAKE_CXX_STANDARD 14)

set(NEAT_GRU_SOURCES
        Train/Train.cpp
        Game/Game.cpp
        Game/Player.cpp
        Private/Generation.cpp
        Private/Generation.h
        Private/Mutation.cpp
        Private/Mutation.h
        Private/Phenotype.cpp
        Private/Phenotype.h
        Private/Species.cpp
        Private/Species.h
        Private/MutationField.cpp
        Private/Random.cpp
        Private/Random.h
        Private/MutationField.h
        NeuralNetwork/Topology.cpp
        Timer.cpp
        Threading/ThreadPool.cpp
        Serializer/Serializable.cpp
        bindings/bindings.cpp
        bindings/GameBinding.cpp
        TopologyParser/TopologyParser.cpp
        Private/Layer.cpp
        )

set(PUB_HEADERS Game/Player.h
        Serializer/Serializable.h
        Serializer/Serializer.hpp
        Threading/ThreadPool.h
        Threading/multithreaded_methods.h
        Timer.h
	TopologyParser/TopologyParser.cpp
        NeuralNetwork/Topology.h
        Game/Game.h
        Train/Train.h
        bindings/bindings.h
        bindings/GameBinding.h
        )

add_library(NEAT_GRU)
target_include_directories(NEAT_GRU PRIVATE Private)
target_include_directories(NEAT_GRU PUBLIC Serializer)
target_include_directories(NEAT_GRU PUBLIC NeuralNetwork)
target_include_directories(NEAT_GRU PUBLIC Game)
target_include_directories(NEAT_GRU PUBLIC Train)
target_include_directories(NEAT_GRU PUBLIC TopologyParser)
target_include_directories(NEAT_GRU PUBLIC Threading)
target_sources(NEAT_GRU PRIVATE ${NEAT_GRU_SOURCES})
target_sources(NEAT_GRU PRIVATE ${PUB_HEADERS})


if (CUDA_ENABLED)
    include_directories(Private/CUDA/NeuralNetwork)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    enable_language(CUDA)
    target_sources(NEAT_GRU PRIVATE
            Private/CUDA/Neuron.cu
            Private/CUDA/Neuron.cuh
            NeuralNetwork/NN.cu
            Private/CUDA/Connection.cu
            Private/CUDA/Connection.cuh
            Private/CUDA/routines.cu
            Private/CUDA/routines.cuh)
    add_definitions(-DCUDA_ENABLED)
    set_target_properties(NEAT_GRU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(NEAT_GRU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_35,code=sm_35>)

else ()
    target_sources(NEAT_GRU PRIVATE
            Private/Neuron.cpp
            Private/Neuron.h
            NeuralNetwork/NN.cpp
            NeuralNetwork/NN.h
            Private/Connection.cpp
            Private/Connection.h
            Private/routines.cpp
            Private/routines.h)
endif ()

target_link_libraries(NEAT_GRU -lstdc++)

INSTALL(TARGETS NEAT_GRU
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

INSTALL(FILES ${PUB_HEADERS}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/neat_gru)
