#include <iostream>

int max(int a, int b) {
    if (a > b) {
        return a;
    }
}

int main() {
    int result = max(10, 20);
    std::cout << "Max: " << result << std::endl;
    return 0;
}