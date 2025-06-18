#include "matrix.hpp"

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<float>(c, 0)) {}

Matrix Matrix::fromVector(const std::vector<float>& vec) {
    Matrix m(vec.size(), 1);
    for (int i = 0; i < vec.size(); ++i) {
        m.data[i][0] = vec[i];
    }
    return m;
}

Matrix Matrix::parallelMultiply(const Matrix& other, int numThreads) const {
    if (cols != other.rows) throw std::invalid_argument("Invalid matrix dimensions for multiplication.");
    Matrix result(rows, other.cols);

    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                float sum = 0;
                for (int k = 0; k < cols; ++k) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
    };

    std::vector<std::thread> threads;
    int chunkSize = rows / numThreads;
    for (int t = 0; t < numThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == numThreads - 1) ? rows : start + chunkSize;
        threads.emplace_back(worker, start, end);
    }
    for (auto& t : threads) t.join();

    return result;
}

void Matrix::print() const {
    for (const auto& row : data) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}
