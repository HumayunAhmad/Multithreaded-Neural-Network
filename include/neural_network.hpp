#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "matrix.hpp"
#include <vector>

class NeuralNetwork {
public:
    std::vector<int> layers;
    std::vector<Matrix> weights;

    NeuralNetwork(std::vector<int> layout);
    Matrix forward(const Matrix& input, int numThreads);
    void train(const std::vector<Matrix>& X, const std::vector<Matrix>& Y, int epochs, float lr, int numThreads);
    Matrix predict(const Matrix& input);
};

#endif
