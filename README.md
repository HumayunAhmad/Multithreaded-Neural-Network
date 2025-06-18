# 🧠 Multithreaded Neural Network in C++

This is a simple neural network implemented from scratch in C++. It supports a configurable architecture and leverages multithreading to speed up matrix operations during training.

## 🚀 Features
- Feedforward neural network architecture
- Custom matrix math implementation
- Multithreaded matrix multiplication using `std::thread`
- Training with sigmoid activation and a basic weight update

## 🧵 How Multithreading is Used
We use multithreading in matrix multiplication for forward propagation. Rows of the output matrix are calculated in parallel, leading to significant speedup in training.

## 🛠️ Build & Run

### Prerequisites
- C++17 compiler
- CMake >= 3.10

### Steps
```bash
mkdir build && cd build
cmake ..
make
./multithreaded_nn
```

## 📊 Example Output
```
0.01 
0.98 
0.97 
0.02 
```

## 🧰 Extend It
- Replace naive gradient step with full backpropagation
- Add ReLU or Softmax activation
- Implement thread pool instead of raw threads
- Load and train on CSV datasets like MNIST
