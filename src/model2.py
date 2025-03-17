import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths in Kaggle
BASE_MODEL_PATH = "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1"  # Base Mistral model
LORA_PATH = "/kaggle/input/llm-daigt-single-best-mistral-7b-lora/transformers/v1/1"  # LoRA adapters

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

# 🔹 Fix: Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use end-of-sequence token as padding

# Quantization configuration for 4-bit inference
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

# Load LoRA configuration
peft_config = PeftConfig.from_pretrained(LORA_PATH)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, LORA_PATH, config=peft_config)

# Move model to device
model.to(device)

# 🔹 Fix: Ensure padding token is set in model config
model.config.pad_token_id = tokenizer.pad_token_id

print("✅ Model loaded successfully!")

def get_ai_generated_score(text, model, tokenizer, device):
    """Returns the AI-generated likelihood score for a given text."""
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Move input tensors to device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract score
    score = torch.sigmoid(outputs.logits).item()  # Convert to probability (0 to 1)

    return score  # Higher means more AI-like
