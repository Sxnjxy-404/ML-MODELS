# Text Classification with Hugging Face Transformers (Complete Project)

A minimal end-to-end project that trains a DistilBERT model for text classification on a CSV file
and serves predictions via a Streamlit app.

## Project Structure
```
text_classification_project/
├── app.py                   # Streamlit inference app
├── train.py                 # Train DistilBERT on your CSV
├── requirements.txt
├── README.md
├── data/
│   └── sample_data.csv      # Tiny sample dataset (text,label)
└── models/
    └── (will be created after training, e.g. ./models/textclf)
```

## 1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

## 2) Install dependencies
```bash
pip install -r requirements.txt
```

## 3) Train
By default, `train.py` uses `./data/sample_data.csv`. Replace with your own CSV having two columns: `text,label`.
```bash
python train.py --csv ./data/sample_data.csv --out_dir ./models/textclf --epochs 3
```

## 4) Run the app
The app looks for a model directory at `./models/textclf` by default. You can also choose any model folder from the UI.
```bash
streamlit run app.py
```

## CSV Format
```csv
text,label
"I love this product!",positive
"This is the worst experience ever.",negative
"Not bad, could be better.",neutral
```

## Notes
- Uses `distilbert-base-uncased` as the default model.
- Saves the trained model to a directory that includes `config.json`, `pytorch_model.bin`, `tokenizer.json`, etc. so it can be loaded by `AutoModelForSequenceClassification` and `AutoTokenizer`.
- If you have a GPU + CUDA set up, PyTorch will use it automatically; otherwise it runs on CPU.
