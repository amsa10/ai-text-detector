# **AI Text Detector – Bulk & Single Detection**  

This project detects AI-generated text using two models:  
1️⃣ **RoBERTa-based classifier (`model1.py`)**  
2️⃣ **LoRA fine-tuned model (`model2.py`)**  

## **📌 Features**  
✅ **Single-text detection** – Quickly check if a short passage is AI-generated.  
✅ **Bulk detection** – Process large datasets (20+ essays) with CSV input.  
✅ **Choose your AI model** – Use **RoBERTa (model1)** or **Mistral LoRA (model2)**.  
✅ **Optimized for efficiency** – Uses a **progress bar** for bulk processing.  

---

## **📌 Setup**  

1️⃣ **Clone the repository**  
```sh
git clone https://github.com/your-username/ai_text_detector.git
cd ai-text-detector
pip install -r requirements.txt
python main.py --model model1
python main.py --model model2

ai_text_detector/
│── data/
│   ├── input.csv  # Your dataset with text to analyze
│   ├── processed_results.csv  # Output results after processing
│── src/
│   ├── model1.py  # RoBERTa-based classifier
│   ├── model2.py  # Mistral 7B + LoRA fine-tuned model
│── requirements.txt  # Dependencies
│── README.md  # Project description and usage guide
│── .gitignore  # Ignore large files, cache, etc.
│── main.py  # Script to run detection on a dataset



