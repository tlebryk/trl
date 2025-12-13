#include <iostream>

void print_value(int* ptr) {
    std::cout << *ptr << std::endl; // Error: potential null dereference
}

int main() {
    int* ptr = nullptr;
    print_value(ptr);
    return 0;
}