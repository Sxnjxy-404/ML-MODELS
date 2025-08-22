#======================================================
# inference_utils.py — Load artifacts and generate replies
#======================================================

import os
import json
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

START = "<start>"
END = "<end>"

# Compiled regex patterns for efficiency
_non_alnum = re.compile(r"[^a-z0-9?'.,! ]+")
_multi_space = re.compile(r"\s+")

def clean(s: str) -> str:
    """Clean text for inference (same as training)"""
    s = str(s).lower().strip()
    s = _non_alnum.sub(" ", s)
    s = _multi_space.sub(" ", s).strip()
    return s

def load_artifacts(model_dir: str):
    """Load saved model, tokenizer, and metadata"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tok = tokenizer_from_json(f.read())
    
    # Load metadata
    meta_path = os.path.join(model_dir, 'meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Load model
    model_path = os.path.join(model_dir, 'seq2seq.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_model(model_path)
    
    print(f"Loaded model from {model_dir}")
    print(f"Vocabulary size: {meta['vocab_size']}")
    print(f"Max sequence length: {meta['max_len']}")
    
    return model, tok, meta

def ids_to_word(tokenizer):
    """Create reverse mapping from token IDs to words"""
    return {idx: w for w, idx in tokenizer.word_index.items()}

def greedy_decode(model, tokenizer, meta, prompt: str, max_new_tokens: int = 30):
    """Generate response using greedy decoding"""
    max_len = int(meta['max_len'])
    w2i = tokenizer.word_index
    i2w = ids_to_word(tokenizer)
    
    # Encode input prompt
    cleaned_prompt = clean(prompt)
    if not cleaned_prompt:
        return "I didn't understand that. Could you rephrase?"
    
    enc = tokenizer.texts_to_sequences([cleaned_prompt])
    if not enc[0]:  # Empty sequence after tokenization
        return "I'm not sure how to respond to that."
    
    enc = pad_sequences(enc, maxlen=max_len, padding='post')
    
    # Start decoder with <start> token
    start_id = w2i.get(START, 0)
    dec_tokens = [start_id]
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Prepare decoder input
        dec_in = pad_sequences([dec_tokens], maxlen=max_len, padding='post')
        
        # Get model prediction
        try:
            preds = model.predict([enc, dec_in], verbose=0)
            # Get the prediction for the last generated token
            next_id = int(np.argmax(preds[0, len(dec_tokens)-1]))
        except Exception as e:
            print(f"Prediction error: {e}")
            break
        
        # Check for end token
        if i2w.get(next_id, '') == END:
            break
        
        # Add predicted token
        dec_tokens.append(next_id)
        
        # Safety check for sequence length
        if len(dec_tokens) >= max_len:
            break
    
    # Convert tokens back to words (skip start token and padding)
    words = []
    for token_id in dec_tokens:
        if token_id == 0:  # padding
            continue
        if token_id == w2i.get(START, -1):  # start token
            continue
        word = i2w.get(token_id, '')
        if word and word != END:
            words.append(word)
    
    response = " ".join(words).strip()
    
    # Fallback response if generation failed
    if not response:
        return "I'm not sure how to respond to that."
    
    return response

def beam_search_decode(model, tokenizer, meta, prompt: str, beam_width: int = 3, max_new_tokens: int = 30):
    """Generate response using beam search (more advanced decoding)"""
    max_len = int(meta['max_len'])
    w2i = tokenizer.word_index
    i2w = ids_to_word(tokenizer)
    
    # Encode input prompt
    cleaned_prompt = clean(prompt)
    if not cleaned_prompt:
        return "I didn't understand that. Could you rephrase?"
    
    enc = tokenizer.texts_to_sequences([cleaned_prompt])
    if not enc[0]:
        return "I'm not sure how to respond to that."
    
    enc = pad_sequences(enc, maxlen=max_len, padding='post')
    
    # Initialize beam with start token
    start_id = w2i.get(START, 0)
    beams = [([start_id], 0.0)]  # (sequence, log_prob)
    
    for _ in range(max_new_tokens):
        candidates = []
        
        for seq, score in beams:
            if i2w.get(seq[-1], '') == END:
                candidates.append((seq, score))
                continue
            
            # Prepare decoder input
            dec_in = pad_sequences([seq], maxlen=max_len, padding='post')
            
            try:
                preds = model.predict([enc, dec_in], verbose=0)
                log_probs = np.log(preds[0, len(seq)-1] + 1e-8)
                
                # Get top beam_width candidates
                top_indices = np.argsort(log_probs)[-beam_width:]
                
                for idx in top_indices:
                    new_seq = seq + [idx]
                    new_score = score + log_probs[idx]
                    candidates.append((new_seq, new_score))
                    
            except Exception as e:
                print(f"Beam search error: {e}")
                candidates.append((seq, score))
        
        # Keep top beam_width candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Check if all beams ended
        if all(i2w.get(seq[-1], '') == END for seq, _ in beams):
            break
    
    # Get best sequence
    best_seq = beams[0][0]
    
    # Convert to words
    words = []
    for token_id in best_seq:
        if token_id == 0 or token_id == w2i.get(START, -1):
            continue
        word = i2w.get(token_id, '')
        if word and word != END:
            words.append(word)
    
    response = " ".join(words).strip()
    return response if response else "I'm not sure how to respond to that."