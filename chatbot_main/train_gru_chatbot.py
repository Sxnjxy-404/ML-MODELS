#======================================================
# train_gru_chatbot.py — GRU Seq2Seq Training Script
#======================================================
# Usage:
# python train_gru_chatbot.py --data_path sample_pairs.csv \
#   --model_dir artifacts --epochs 20 --max_vocab 12000 --max_len 32
#======================================================

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense

# ----------------------
# Text cleaning
# ----------------------
START_TOKEN = "<start>"
END_TOKEN = "<end>"

def clean_text(s: str) -> str:
    """Clean and normalize text"""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9?'.,! ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV or create default data"""
    if not os.path.exists(path):
        print(f"File {path} not found. Creating default dataset...")
        data = {
            "prompt": [
                "hi", "hello", "how are you", "what is ai", "what is machine learning",
                "tell me a joke", "thanks", "bye"
            ],
            "reply": [
                "hello! how can i help you today?",
                "hi there! what brings you here?",
                "i'm doing great, thanks for asking!",
                "ai is the science of making machines intelligent.",
                "machine learning lets computers learn patterns from data.",
                "why did the model cross the road? to optimize the other side!",
                "you're welcome!",
                "goodbye! have a great day!"
            ]
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)
    
    # Clean and prepare data
    df["prompt"] = df["prompt"].apply(clean_text)
    df["reply"] = df["reply"].apply(lambda s: f"{START_TOKEN} " + clean_text(s) + f" {END_TOKEN}")
    
    print(f"Loaded {len(df)} conversation pairs")
    return df

def vectorize(df: pd.DataFrame, max_vocab: int = 12000, max_len: int = 32):
    """Convert text to sequences and prepare training data"""
    # Create tokenizer on all text
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<oov>")
    all_texts = pd.concat([df["prompt"], df["reply"]], axis=0)
    tokenizer.fit_on_texts(all_texts)
    
    # Encode inputs (prompts)
    enc_in = tokenizer.texts_to_sequences(df["prompt"].tolist())
    enc_in = pad_sequences(enc_in, maxlen=max_len, padding="post")
    
    # Encode outputs (replies)
    dec_full = tokenizer.texts_to_sequences(df["reply"].tolist())
    
    # Decoder input: all tokens except last
    dec_in = [seq[:-1] for seq in dec_full]
    # Decoder target: all tokens except first
    dec_tar = [seq[1:] for seq in dec_full]
    
    dec_in = pad_sequences(dec_in, maxlen=max_len, padding="post")
    dec_tar = pad_sequences(dec_tar, maxlen=max_len, padding="post")
    
    vocab_size = min(max_vocab, len(tokenizer.word_index) + 1)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Encoder input shape: {enc_in.shape}")
    print(f"Decoder input shape: {dec_in.shape}")
    print(f"Decoder target shape: {dec_tar.shape}")
    
    return enc_in, dec_in, dec_tar, tokenizer, vocab_size

def build_train_model(vocab_size: int, emb_dim: int, hid_dim: int, max_len: int):
    """Build encoder-decoder model for training"""
    
    # Input layers
    enc_inputs = Input(shape=(max_len,), name="encoder_inputs")
    dec_inputs = Input(shape=(max_len,), name="decoder_inputs")
    
    # Shared embedding layer
    emb = Embedding(vocab_size, emb_dim, mask_zero=True, name="shared_embedding")
    
    # Encoder
    enc_emb = emb(enc_inputs)
    enc_gru = GRU(hid_dim, return_state=True, name="encoder_gru")
    _, enc_state = enc_gru(enc_emb)
    
    # Decoder (with teacher forcing)
    dec_emb = emb(dec_inputs)
    dec_gru = GRU(hid_dim, return_sequences=True, return_state=True, name="decoder_gru")
    dec_outputs, _ = dec_gru(dec_emb, initial_state=enc_state)
    
    # Output layer
    dec_dense = Dense(vocab_size, activation='softmax', name="output_dense")
    outputs = dec_dense(dec_outputs)
    
    # Create and compile model
    model = Model([enc_inputs, dec_inputs], outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def save_artifacts(model, tokenizer, model_dir: str):
    """Save trained model and tokenizer"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model.save(os.path.join(model_dir, "seq2seq.h5"))
    
    # Save tokenizer
    with open(os.path.join(model_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
    
    print(f"Model and tokenizer saved to {model_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train GRU Chatbot")
    parser.add_argument('--data_path', type=str, default='sample_pairs.csv',
                        help='Path to CSV file with prompt,reply columns')
    parser.add_argument('--model_dir', type=str, default='artifacts',
                        help='Directory to save model artifacts')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--max_vocab', type=int, default=12000,
                        help='Maximum vocabulary size')
    parser.add_argument('--max_len', type=int, default=32,
                        help='Maximum sequence length')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--hid_dim', type=int, default=256,
                        help='Hidden state dimension')
    
    args = parser.parse_args()
    
    print("Loading and preprocessing data...")
    df = load_dataset(args.data_path)
    
    print("Vectorizing text...")
    enc_in, dec_in, dec_tar, tokenizer, vocab_size = vectorize(
        df, args.max_vocab, args.max_len
    )
    
    print("Building model...")
    model = build_train_model(vocab_size, args.emb_dim, args.hid_dim, args.max_len)
    model.summary()
    
    print("Starting training...")
    history = model.fit(
        [enc_in, dec_in],
        dec_tar[..., None],  # sparse targets need extra dimension
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    print("Saving artifacts...")
    save_artifacts(model, tokenizer, args.model_dir)
    
    # Save training metadata
    meta = {
        "vocab_size": vocab_size,
        "max_len": int(args.max_len),
        "emb_dim": int(args.emb_dim),
        "hid_dim": int(args.hid_dim),
        "start_token": START_TOKEN,
        "end_token": END_TOKEN,
    }
    
    with open(os.path.join(args.model_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"Training complete! All artifacts saved to: {args.model_dir}")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    if 'val_accuracy' in history.history:
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

if __name__ == '__main__':
    main()