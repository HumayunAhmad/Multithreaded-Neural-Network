#include "matrix.hpp"
#include "neural_network.hpp"

int main() {
    NeuralNetwork net({2, 4, 1});
    std::vector<Matrix> X = {
        Matrix::fromVector({0, 0}),
        Matrix::fromVector({0, 1}),
        Matrix::fromVector({1, 0}),
        Matrix::fromVector({1, 1})
    };

    std::vector<Matrix> Y = {
        Matrix::fromVector({0}),
        Matrix::fromVector({1}),
        Matrix::fromVector({1}),
        Matrix::fromVector({0})
    };

    net.train(X, Y, 1000, 0.1, 4);

    for (auto& x : X) {
        Matrix pred = net.predict(x);
        pred.print();
    }

    return 0;
}
