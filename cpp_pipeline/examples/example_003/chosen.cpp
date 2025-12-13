#include <iostream>

int square(int x) {
    return x * x;
}

int main() {
    int num = 5;
    int result = square(num);
    std::cout << "Square of " << num << " is " << result << std::endl;
    return 0;
}