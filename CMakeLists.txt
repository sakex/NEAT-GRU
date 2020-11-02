cmake_minimum_required(VERSION 3.14)

project(neat_gru)
SET(CMAKE_CXX_STANDARD 17)


if (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
endif ()

SET(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
if(!__EMSCRIPTEN__)
SET(CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -ffast-math -frename-registers -flto")
else()
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -flto -s ENVIRONMENT='web'")
endif()

if (CMAKE_BUILD_TYPE STREQUAL Release)
    ADD_COMPILE_DEFINITIONS(__MULTITHREADED__=1)
endif ()


SET(NEAT_GRU_SOURCES
        Train/Train.cpp
        Game/Game.cpp
        Private/Generation.cpp
        Private/Mutation.cpp
        Private/Gene.cpp
        Private/Species.cpp
        Private/MutationField.cpp
        Private/Random.cpp
        NeuralNetwork/Topology.cpp
        Timer.cpp
        Serializer/Serializable.cpp
        bindings/bindings.cpp
        bindings/GameBinding.cpp
        TopologyParser/TopologyParser.cpp
        )

SET(PUB_HEADERS
        Serializer/Serializable.h
        Serializer/Serializer.hpp
        Threading/multithreaded_methods.h
        Timer.h
        TopologyParser/TopologyParser.cpp
        NeuralNetwork/Topology.h
        Game/Game.h
        Train/Train.h
        bindings/bindings.h
        bindings/GameBinding.h
        nlohmann/json.hpp
        )

ADD_LIBRARY(NEAT_GRU Train/static.h)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PRIVATE Private)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC Serializer)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC NeuralNetwork)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC Game)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC Train)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC TopologyParser)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC Threading)
TARGET_INCLUDE_DIRECTORIES(NEAT_GRU PUBLIC nlohmann)
TARGET_SOURCES(NEAT_GRU PRIVATE ${NEAT_GRU_SOURCES})

IF (CUDA_ENABLED)
    MESSAGE(WARNING CUDA_ENABLED)
    ADD_COMPILE_DEFINITIONS(CUDA_ENABLED=1)
    ADD_SUBDIRECTORY(GPU)
    TARGET_LINK_LIBRARIES(NEAT_GRU NEAT_CUDA)
ENDIF ()
ADD_COMPILE_OPTIONS(-Wall -Wextra -pedantic -Werror)
TARGET_SOURCES(NEAT_GRU PRIVATE
        NeuralNetwork/NN.cpp
        )
SET(PUB_HEADERS ${PUB_HEADERS}
        NeuralNetwork/NN.h
        NeuralNetwork/ConnectionGru.h
        NeuralNetwork/Neuron.h
        )

TARGET_SOURCES(NEAT_GRU PRIVATE ${PUB_HEADERS})


TARGET_LINK_LIBRARIES(NEAT_GRU -lstdc++)

if (CMAKE_PROJECT_NAME STREQUAL TEST)
    include(CTest)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(tests)
endif ()


INSTALL(TARGETS NEAT_GRU
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

INSTALL(FILES ${PUB_HEADERS}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/neat_gru)
