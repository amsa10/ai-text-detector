import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths (Modify these if not using Kaggle)
BASE_MODEL_PATH = "mistral/pytorch/7b-v0.1-hf/1"  # Base Mistral model
LORA_PATH = "llm-daigt-single-best-mistral-7b-lora/transformers/v1/1"  # LoRA adapters

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# Ensure tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=1,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Ensure model has a classification head
if not hasattr(model, "score"):
    model.score = nn.Linear(model.config.hidden_size, 1).to(device)

# Load LoRA adapters
peft_config = PeftConfig.from_pretrained(LORA_PATH)
model = PeftModel.from_pretrained(model, LORA_PATH, config=peft_config)

# Move model to device
model.to(device)

# Ensure padding token is set
model.config.pad_token_id = tokenizer.pad_token_id

print("✅ Model loaded successfully!")


def get_ai_generated_score(text):
    """Returns the AI-generated likelihood score for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    score = torch.sigmoid(outputs.logits).item()  # Convert to probability
    return score  # Higher = more AI-like


def process_bulk_data(input_file, output_file):
    """Process bulk text data using model2."""
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(input_file)
    tqdm.pandas()
    df["scores"] = df["text"].progress_apply(get_ai_generated_score)
    df.to_csv(output_file, index=False)
    print(f"✅ Processed results saved to {output_file}")

