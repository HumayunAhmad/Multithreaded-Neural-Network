cmake_minimum_required(VERSION 3.10)
project(MultithreadedNeuralNet)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(multithreaded_nn
    src/main.cpp
    src/matrix.cpp
    src/neural_network.cpp
)
