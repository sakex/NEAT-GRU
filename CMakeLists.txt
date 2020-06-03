project(neat_gru)
cmake_minimum_required(VERSION 3.0)
if (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
endif ()

SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -ffast-math -frename-registers -flto")
SET(CMAKE_CXX_STANDARD 14)

if (CMAKE_BUILD_TYPE STREQUAL Release)
    add_compile_definitions(__MULTITHREADED__=1)
endif ()


SET(NEAT_GRU_SOURCES
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
        Private/Neuron.cpp
        Private/Neuron.h
        NeuralNetwork/NN.cpp
        NeuralNetwork/NN.h
        Private/Connection.cpp
        Private/Connection.h
        Private/routines.cpp
        Private/routines.h
        )

SET(PUB_HEADERS Game/Player.h
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

ADD_LIBRARY(NEAT_GRU Train/static.h)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PRIVATE Private)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC Serializer)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC NeuralNetwork)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC Game)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC Train)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC TopologyParser)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC Threading)
TARGET_SOURCES(NEAT_GRU PRIVATE ${NEAT_GRU_SOURCES})
TARGET_SOURCES(NEAT_GRU PRIVATE ${PUB_HEADERS})


if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif ()

ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)
ADD_TEST(neat_gru neat_gru_testing)

TARGET_LINK_LIBRARIES(NEAT_GRU -lstdc++)


INSTALL(TARGETS NEAT_GRU
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

INSTALL(FILES ${PUB_HEADERS}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/neat_gru)
