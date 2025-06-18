#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <thread>
#include <functional>
#include <stdexcept>
#include <iostream>

class Matrix {
public:
    int rows, cols;
    std::vector<std::vector<float>> data;

    Matrix(int r, int c);
    static Matrix fromVector(const std::vector<float>& vec);
    Matrix parallelMultiply(const Matrix& other, int numThreads) const;
    void print() const;
};

#endif
