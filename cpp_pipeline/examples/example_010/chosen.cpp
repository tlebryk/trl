#include <iostream>

void print_value(int* ptr) {
    if (ptr != nullptr) {
        std::cout << *ptr << std::endl;
    } else {
        std::cout << "Pointer is null" << std::endl;
    }
}

int main() {
    int* ptr = nullptr;
    print_value(ptr);
    return 0;
}