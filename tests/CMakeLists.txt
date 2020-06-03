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