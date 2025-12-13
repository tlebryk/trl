#include <iostream>

int safe_divide(int a, int b) {
    if (b == 0) {
        return 0; // Handle safely
    }
    return a / b;
}

int main() {
    std::cout << safe_divide(10, 0) << std::endl;
    return 0;
}