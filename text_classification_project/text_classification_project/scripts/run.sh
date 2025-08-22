
#!/usr/bin/env bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python train.py --csv ./data/sample_data.csv --out_dir ./models/textclf --epochs 1
streamlit run app.py
