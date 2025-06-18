#include "neural_network.hpp"
#include <random>
#include <cmath>

NeuralNetwork::NeuralNetwork(std::vector<int> layout) : layers(layout) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1, 1);
    for (size_t i = 1; i < layers.size(); ++i) {
        Matrix w(layers[i], layers[i - 1]);
        for (auto& row : w.data)
            for (auto& val : row)
                val = dist(gen);
        weights.push_back(w);
    }
}

Matrix NeuralNetwork::forward(const Matrix& input, int numThreads) {
    Matrix out = input;
    for (const auto& w : weights) {
        out = w.parallelMultiply(out, numThreads);
        for (auto& row : out.data)
            for (auto& val : row)
                val = 1 / (1 + std::exp(-val)); // sigmoid
    }
    return out;
}

void NeuralNetwork::train(const std::vector<Matrix>& X, const std::vector<Matrix>& Y, int epochs, float lr, int numThreads) {
    for (int e = 0; e < epochs; ++e) {
        for (size_t i = 0; i < X.size(); ++i) {
            Matrix pred = forward(X[i], numThreads);
            for (int j = 0; j < pred.rows; ++j) {
                float error = Y[i].data[j][0] - pred.data[j][0];
                for (auto& row : weights.back().data)
                    for (auto& val : row)
                        val += lr * error; // Naive weight update
            }
        }
    }
}

Matrix NeuralNetwork::predict(const Matrix& input) {
    return forward(input, 4);
}
