import json
import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DEFAULT_MODEL_DIR = "./models/textclf"

@st.cache_resource(show_spinner=False)
def load_pipeline(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    clf = pipeline("text-classification", model=model, tokenizer=tok, return_all_scores=True)
    label_names = None
    label_path = os.path.join(model_dir, "label_names.json")
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            label_names = json.load(f)
    return clf, label_names

st.set_page_config(page_title="Text Classifier", page_icon="🤖")
st.title("🤖 Text Classification (Hugging Face)")

st.markdown("""
- Choose a model directory (default: `./models/textclf`).
- Enter text(s) to classify.
""")

model_dir = st.text_input("Model directory", value=DEFAULT_MODEL_DIR)
load_btn = st.button("Load model")
if load_btn or "clf" not in st.session_state:
    try:
        clf, label_names = load_pipeline(model_dir)
        st.session_state["clf"] = clf
        st.session_state["label_names"] = label_names
        st.success(f"Loaded model from: {model_dir}")
    except Exception as e:
        st.error(f"Failed to load model from {model_dir}: {e}")

text = st.text_area("Enter text (one per line for batch):", height=160, value="I love this product!\nThis is the worst experience ever.")
run = st.button("Classify")

if run and "clf" in st.session_state:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    results = st.session_state["clf"](lines)
    show_names = st.session_state.get("label_names")

    for inp, scores in zip(lines, results):
        st.subheader(inp)
        scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        for s in scores:
            label = s["label"]
            if show_names and label.startswith("LABEL_"):
                try:
                    idx = int(label.split("_")[-1])
                    label = show_names[idx]
                except:
                    pass
            st.write(f"- **{label}**: {s['score']:.4f}")
        st.markdown("---")
