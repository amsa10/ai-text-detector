# **AI Text Detector – Bulk & Single Detection**  

This project detects AI-generated text using a **RoBERTa-based classifier**.  

## **📌 Features**  
✅ **Single-text detection** – Quickly check if a short passage is AI-generated.  
✅ **Bulk detection** – Process large datasets (20+ essays) with CSV input.  
✅ **Optimized for efficiency** – Uses a **progress bar** for bulk processing.  

---

## **📌 Setup**  

1️⃣ **Clone the repository**  
```sh
git clone https://github.com/your-username/ai_text_detector.git
cd ai-text-detector
python main.py

ai_text_detector/
│── data/
│   ├── input.csv  # Your dataset with text to analyze
│   ├── processed_results.csv  # Output results after processing
│── src/
│   ├── model1.py  # AI detection logic RoBERTa-based classifier
│── requirements.txt  # Dependencies
│── README.md  # Project description and usage guide
│── .gitignore  # Ignore large files, cache, etc.
│── main.py  # Script to run detection on a dataset


