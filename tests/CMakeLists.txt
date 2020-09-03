ADD_EXECUTABLE(neat_gru_testing
        MemoryGame/test_main.cpp
        MemoryGame/Memory.cpp
        MemoryGame/Memory.h
        MemoryGame/MemoryPlayer.cpp
        MemoryGame/MemoryPlayer.h
        MemoryGame/MemoryGrid.cpp
        MemoryGame/MemoryGrid.h
        MemoryGame/constants.h)

TARGET_LINK_LIBRARIES(neat_gru_testing NEAT_GRU pthread)

ADD_TEST(neat_gru neat_gru_testing)

ADD_EXECUTABLE(neat_trend_testing
        TrendGame/test_main.cpp
        TrendGame/TrendGame.cpp
        TrendGame/TrendGame.h
        )

ADD_TEST(neat_gru neat_trend_testing)
TARGET_LINK_LIBRARIES(neat_trend_testing NEAT_CUDA pthread)

ADD_EXECUTABLE(gpu_test
        gpu/gpu_main.cpp
        gpu/GpuGame.cpp
        )

ADD_TEST(neat_gru gpu_test)
TARGET_LINK_LIBRARIES(gpu_test NEAT_GRU pthread)


ADD_EXECUTABLE(test_routines test_routines.cpp)

ADD_TEST(neat_gru test_routines)
TARGET_LINK_LIBRARIES(test_routines NEAT_GRU)
