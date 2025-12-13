#include <iostream>

double divide(double a, double b) {
    if (b == 0.0) {
        return 0.0;
    }
    return a / b;
}

int main() {
    double result = divide(10.0, 3.0);
    std::cout << "Result: " << result << std::endl;
    return 0;
}