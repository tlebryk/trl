#include <iostream>

int safe_divide(int a, int b) {
    return a / b; // Error: Division by zero when b=0
}

int main() {
    std::cout << safe_divide(10, 0) << std::endl;
    return 0;
}