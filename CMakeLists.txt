if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -ffast-math -frename-registers -fopenmp -flto")
set(CMAKE_CXX_STANDARD 14)

set(NEAT_GRU_SOURCES
        Train/Train.cpp
        Train/Train.h
        Game/Game.cpp
        Game/Game.h
        Game/Player.cpp
        Game/Player.h
        Private/Generation.cpp
        Private/Generation.h
        Private/Mutation.cpp
        Private/Mutation.h
        Private/Phenotype.cpp
        Private/Phenotype.h
        Private/Species.cpp
        Private/Species.h
        Private/MutationField.cpp
        Private/MutationField.h
        Private/Random.cpp
        Private/Random.h
        NeuralNetwork/Topology.cpp
        NeuralNetwork/Topology.h
        Timer.cpp
        Timer.h
        TopologyParser/TopologyParser.cpp
        TopologyParser/TopologyParser.h
        Threading/multithreaded_methods.h
        Threading/ThreadPool.cpp
        Threading/ThreadPool.h
        Serializer/Serializable.cpp
        Serializer/Serializable.h
        Serializer/Serializer.hpp
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

if (CUDA_ENABLED)
    include_directories(Private/CUDA/NeuralNetwork)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    enable_language(CUDA)
    target_sources(NEAT_GRU PRIVATE
            Private/CUDA/Neuron.cu
            Private/CUDA/Neuron.cuh
            Private/CUDA/NN.cu
            Private/CUDA/NN.h
            Private/CUDA/Connection.cu
            Private/CUDA/Connection.cuh
            Private/CUDA/routines.cu
            Private/CUDA/routines.cuh)
    set_target_properties(trading PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(trading PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_35,code=sm_35>)

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
