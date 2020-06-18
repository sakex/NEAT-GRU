project(neat_gpu LANGUAGES CUDA)
FIND_PACKAGE(CUDA REQUIRED)
SET(CMAKE_CUDA_STANDARD 14)

ADD_LIBRARY(NEAT_CUDA
        NN.cu
        Connection.cu
        Neuron.cu
        )

SET(PUB_HEADERS
        NN.cuh
        Connection.cuh
        Neuron.cuh
        )

TARGET_SOURCES(NEAT_CUDA PRIVATE ${PUB_HEADERS})

set_target_properties(NEAT_CUDA PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

string(APPEND CMAKE_CUDA_FLAGS " -rdc=true")

string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52")
string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50")
string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_35,code=sm_35")


set(CUDA_LANG "$<COMPILE_LANGUAGE:CUDA>")

SET_TARGET_PROPERTIES(
        NEAT_CUDA
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

TARGET_LINK_LIBRARIES(NEAT_CUDA ${CUDA_LIBRARIES} -lcudadevrt)
