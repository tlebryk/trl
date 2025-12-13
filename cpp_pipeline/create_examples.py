"""
Create C++ code examples and save them to disk.

This module creates example definitions and saves them as:
- examples/{example_id}/chosen.cpp
- examples/{example_id}/rejected.cpp
- examples/{example_id}/metadata.json
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def create_example_definitions() -> List[Dict]:
    """
    Create training examples with C++ code.
    
    Returns:
        List of example dicts with: prompt, chosen, rejected
    """
    examples = []
    
    # Example 1: Simple function with missing semicolon
    examples.append({
        "prompt": "Write a C++ function that adds two integers and returns the result.",
        "chosen": """#include <iostream>

int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);
    std::cout << "Result: " << result << std::endl;
    return 0;
}""",
        "rejected": """#include <iostream>

int add(int a, int b) {
    return a + b
}

int main() {
    int result = add(5, 3);
    std::cout << "Result: " << result << std::endl;
    return 0;
}""",
    })
    
    # Example 2: Missing include
    examples.append({
        "prompt": "Write a C++ program that prints 'Hello, World!'",
        "chosen": """#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}""",
        "rejected": """int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}""",
    })
    
    # Example 3: Undeclared variable
    examples.append({
        "prompt": "Write a C++ function that calculates the square of a number.",
        "chosen": """#include <iostream>

int square(int x) {
    return x * x;
}

int main() {
    int num = 5;
    int result = square(num);
    std::cout << "Square of " << num << " is " << result << std::endl;
    return 0;
}""",
        "rejected": """#include <iostream>

int square(int x) {
    return x * x;
}

int main() {
    result = square(5);
    std::cout << "Square is " << result << std::endl;
    return 0;
}""",
    })
    
    # Example 4: Type mismatch
    examples.append({
        "prompt": "Write a C++ program that divides two numbers.",
        "chosen": """#include <iostream>

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
}""",
        "rejected": """#include <iostream>

int divide(int a, int b) {
    return a / b;
}

int main() {
    double result = divide(10.0, 3.0);
    std::cout << "Result: " << result << std::endl;
    return 0;
}""",
    })
    
    # Example 5: Missing return statement
    examples.append({
        "prompt": "Write a C++ function that returns the maximum of two integers.",
        "chosen": """#include <iostream>

int max(int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}

int main() {
    int result = max(10, 20);
    std::cout << "Max: " << result << std::endl;
    return 0;
}""",
        "rejected": """#include <iostream>

int max(int a, int b) {
    if (a > b) {
        return a;
    }
}

int main() {
    int result = max(10, 20);
    std::cout << "Max: " << result << std::endl;
    return 0;
}""",
    })
    
    # Example 6: Syntax error - unmatched braces
    examples.append({
        "prompt": "Write a C++ program that uses a for loop to print numbers 1 to 10.",
        "chosen": """#include <iostream>

int main() {
    for (int i = 1; i <= 10; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    return 0;
}""",
        "rejected": """#include <iostream>

int main() {
    for (int i = 1; i <= 10; i++) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    return 0;
""",
    })
    
    # Example 7: Using namespace (C++ specific)
    examples.append({
        "prompt": "Write a C++ program using std::vector to store and print numbers.",
        "chosen": """#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return 0;
}""",
        "rejected": """#include <iostream>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return 0;
}""",
    })

    # Example 8: Runtime Error - Buffer Overflow
    examples.append({
        "prompt": "Write a C++ program that creates an array of 5 integers and accesses the last element.",
        "chosen": """#include <iostream>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    std::cout << "Last element: " << arr[4] << std::endl;
    return 0;
}""",
        "rejected": """#include <iostream>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    // Error: Accessing out of bounds (index 5 is the 6th element)
    std::cout << "Last element: " << arr[5] << std::endl;
    return 0;
}""",
    })

    # Example 9: Runtime Error - Division by Zero
    examples.append({
        "prompt": "Write a C++ function to divide two integers safely.",
        "chosen": """#include <iostream>

int safe_divide(int a, int b) {
    if (b == 0) {
        return 0; // Handle safely
    }
    return a / b;
}

int main() {
    std::cout << safe_divide(10, 0) << std::endl;
    return 0;
}""",
        "rejected": """#include <iostream>

int safe_divide(int a, int b) {
    return a / b; // Error: Division by zero when b=0
}

int main() {
    std::cout << safe_divide(10, 0) << std::endl;
    return 0;
}""",
    })
    
    # Example 10: Runtime Error - Null Pointer Dereference
    examples.append({
        "prompt": "Write a C++ program that dereferences a pointer safely.",
        "chosen": """#include <iostream>

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
}""",
        "rejected": """#include <iostream>

void print_value(int* ptr) {
    std::cout << *ptr << std::endl; // Error: potential null dereference
}

int main() {
    int* ptr = nullptr;
    print_value(ptr);
    return 0;
}""",
    })
    
    return examples


def save_example(example: Dict, example_id: str, base_dir: str = "cpp_pipeline/examples"):
    """
    Save an example to disk.
    
    Args:
        example: Dict with prompt, chosen, rejected
        example_id: Unique identifier (e.g., "example_001")
        base_dir: Base directory for examples
    """
    example_dir = Path(base_dir) / example_id
    example_dir.mkdir(parents=True, exist_ok=True)
    
    # Save code files
    (example_dir / "chosen.cpp").write_text(example["chosen"])
    (example_dir / "rejected.cpp").write_text(example["rejected"])
    
    # Save metadata
    metadata = {
        "example_id": example_id,
        "prompt": example["prompt"],
        "created_at": datetime.now().isoformat(),
        "source": "manual",  # or "model_generated" for online RL
        "model_version": None,  # For online RL: track which model generated this
    }
    
    (example_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


def create_all_examples(base_dir: str = "cpp_pipeline/examples"):
    """
    Create all examples and save them to disk.
    
    Returns:
        List of example IDs created
    """
    examples = create_example_definitions()
    example_ids = []
    
    for idx, example in enumerate(examples, 1):
        example_id = f"example_{idx:03d}"
        save_example(example, example_id, base_dir)
        example_ids.append(example_id)
        print(f"Created {example_id}")
    
    return example_ids


if __name__ == "__main__":
    print("Creating C++ code examples...")
    example_ids = create_all_examples()
    print(f"\nCreated {len(example_ids)} examples:")
    for eid in example_ids:
        print(f"  - {eid}")

