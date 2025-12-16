#include <iostream>

int divide(int a, int b) {
    return a / b;
}

int main() {
    double result = divide(10.0, 3.0);
    std::cout << "Result: " << result << std::endl;
    return 0;
}