#include <iostream>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    // Error: Accessing out of bounds (index 5 is the 6th element)
    std::cout << "Last element: " << arr[5] << std::endl;
    return 0;
}