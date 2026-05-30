# prepara_dataset.py
import json
from datasets import load_dataset

def main():
    print("Downloading saralazza/llada-safety-dataset in progress...")
    # Load the 'test' split of the dataset
    dataset = load_dataset("saralazza/llada-safety-dataset", split="test")
    
    formatted_data = []
    
    for item in dataset:
        source_dataset = item.get("source_dataset", "")
        source_lower = source_dataset.lower()
        
        # Map fields for jailbreakbench_llada.py
        # Save the source in "source" so it gets automatically copied to the output JSON
        formatted_item = {
            "vanilla prompt": item.get("behavior", ""),
            "refined prompt": item.get("refined_behavior", ""),
            "source": source_dataset  
        }
        
        formatted_data.append(formatted_item)
        
    # The filename MUST contain the word "refine" to activate the internal logic of the code
    output_filename = "llada_safety_test_refine.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nDataset prepared and saved in: {output_filename}")
    print(f"Total elements extracted: {len(formatted_data)}")
    safe_count = sum(1 for x in formatted_data if "alpaca" in x["source"].lower())
    print(f"Of which Safe (source contains 'alpaca'): {safe_count}")
    print(f"Of which Unsafe (other sources): {len(formatted_data) - safe_count}")

if __name__ == "__main__":
    main()