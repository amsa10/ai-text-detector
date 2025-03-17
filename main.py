from src.model1 import process_bulk_data

# Define file paths
INPUT_FILE = "data/input.csv"
OUTPUT_FILE = "data/processed_results.csv"

if __name__ == "__main__":
    process_bulk_data(INPUT_FILE, OUTPUT_FILE)
