ADD_EXECUTABLE(neat_gru_testing
        test_main.cpp
        Memory.cpp
        Memory.h
        MemoryPlayer.cpp
        MemoryPlayer.h
        MemoryGrid.cpp
        MemoryGrid.h
        constants.h)

TARGET_LINK_LIBRARIES(neat_gru_testing NEAT_GRU pthread)