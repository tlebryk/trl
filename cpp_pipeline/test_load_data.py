from cpp_pipeline.load_data import get_multipl_e_dataset

def test_load_data():
    print("Loading MultiPL-E dataset (train split)...")
    examples = get_multipl_e_dataset("train")
    print(f"Loaded {len(examples)} examples.")
    
    if len(examples) > 0:
        ex = examples[0]
        print("\nExample 0:")
        print("KEYS:", ex.keys())
        print("PROMPT (first 50 chars):", ex["prompt"][:50])
        print("TESTS (first 50 chars):", ex["tests"][:50])

if __name__ == "__main__":
    test_load_data()
