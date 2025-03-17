import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
MODEL_NAME = "roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def detect_ai_text(text):
    """Detect if the given text is AI-generated."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        return probabilities[:, 1].item()  # Probability of AI-generated text

def process_bulk_data(input_file, output_file):
    """Process bulk text data from a CSV file."""
    df = pd.read_csv(input_file)
    tqdm.pandas()  # Enable progress bar
    df["scores"] = df["text"].progress_apply(detect_ai_text)
    df.to_csv(output_file, index=False)
    print(f"✅ Processed results saved to {output_file}")
