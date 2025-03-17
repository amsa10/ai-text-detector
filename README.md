AI Text Detector – Bulk & Single Detection
This project detects AI-generated text using a RoBERTa-based classifier.

📌 Features
✅ Single-text detection – Quickly check if a short passage is AI-generated.
✅ Bulk detection – Process large datasets (100+ essays) with CSV input.
✅ Optimized for efficiency – Uses a progress bar for bulk processing.

📌 Setup
1️⃣ Clone the repository

sh
Копировать
Редактировать
git clone https://github.com/your-username/ai-text-detector.git
cd ai-text-detector
2️⃣ Install dependencies

sh
Копировать
Редактировать
pip install -r requirements.txt
3️⃣ Run the AI detection script

Single text input

python
Копировать
Редактировать
from src.detector import detect_ai_text

text = "This is an AI-generated essay."
probability = detect_ai_text(text)
print(f"AI-generated probability: {probability:.2f}")
Bulk detection (100+ essays)

sh
Копировать
Редактировать
python main.py
This processes data/input.csv and saves results in data/processed_results.csv.

📂 Project Structure
graphql
Копировать
Редактировать
ai_text_detector_project/
│── data/
│   ├── input.csv  # Your dataset with text to analyze
│   ├── processed_results.csv  # Output results after processing
│── src/
│   ├── detector.py  # AI detection logic
│── notebooks/
│   ├── analysis.ipynb  # Jupyter notebook for experiments
│── requirements.txt  # Dependencies
│── README.md  # Project description and usage guide
│── .gitignore  # Ignore large files, cache, etc.
│── main.py  # Script to run detection on a dataset
📊 Example Output
id	text	ai_generated
1	"This is a simple sentence written by a human."	0.02
2	"The evolution of artificial intelligence..."	0.89
3	"Hello! How can I assist you today?"	0.15
📢 Contributing
Feel free to open issues or pull requests to improve this project! 🚀
