from cpp_pipeline.dataset_utils import load_and_split_dataset, get_prompt_and_tests, stitch_solution

def test_dataset_structure():
    print("Loading and splitting dataset...")
    train_ds, test_ds = load_and_split_dataset()
    
    # Take the first example from train
    example = train_ds[0]
    prompt, tests = get_prompt_and_tests(example)
    
    print("\n--- PROMPT ---")
    print(prompt)
    print("\n--- TESTS ---")
    print(tests)
    
    # Simulate a dummy solution
    dummy_solution = "    // This is where the generated code would go\n    return true;"
    
    full_code = stitch_solution(prompt, dummy_solution, tests)
    
    print("\n--- FULL STITCHED CODE (PREVIEW) ---")
    print(full_code)

if __name__ == "__main__":
    test_dataset_structure()
