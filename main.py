import argparse

# Define file paths
INPUT_FILE = "data/input.csv"
OUTPUT_FILE = "data/processed_results.csv"

def main(model_choice):
    if model_choice == "model1":
        from src.model1 import process_bulk_data
    elif model_choice == "model2":
        from src.model2 import process_bulk_data
    else:
        raise ValueError("Invalid model choice! Use 'model1' or 'model2'.")
    
    process_bulk_data(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the AI detection model.")
    parser.add_argument("--model", choices=["model1", "model2"], required=True, help="Select 'model1' or 'model2'")
    args = parser.parse_args()

    main(args.model)

