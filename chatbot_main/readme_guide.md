# 🤖 GRU Chatbot

A sequence-to-sequence chatbot implementation using GRU (Gated Recurrent Unit) neural networks with TensorFlow/Keras. This project includes training scripts, inference utilities, and both Streamlit and Flask interfaces.

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
python setup.py
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train_gru_chatbot.py

# 3. Run Streamlit interface
streamlit run chatbot_streamlit.py
```

## 📁 Project Structure

```
gru-chatbot/
├── 📄 sample_pairs.csv          # Training data (prompt,reply pairs)
├── 📄 requirements.txt          # Python dependencies
├── 🎓 train_gru_chatbot.py     # Model training script
├── 🔧 inference_utils.py       # Inference utilities
├── 🌐 chatbot_streamlit.py     # Streamlit web interface
├── 🔌 chatbot_flask.py         # Flask REST API
├── ⚙️ setup.py                 # Automated setup script
├── 🧪 run_examples.py          # Usage examples & testing
└── 📁 artifacts/               # Generated model files
    ├── seq2seq.h5              # Trained model
    ├── tokenizer.json          # Text tokenizer
    └── meta.json               # Model metadata
```

## 🎯 Features

- **GRU Seq2Seq Architecture**: Encoder-decoder model with attention
- **Flexible Training**: Easy to retrain with your own conversation data
- **Multiple Interfaces**: Web UI (Streamlit) and REST API (Flask)
- **Advanced Decoding**: Both greedy and beam search decoding
- **Production Ready**: Error handling, validation, and performance monitoring

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies
```bash
pip install tensorflow>=2.12 pandas numpy flask streamlit
```

## 🎓 Training

### Basic Training
```bash
python train_gru_chatbot.py
```

### Advanced Training Options
```bash
python train_gru_chatbot.py \
  --data_path my_data.csv \
  --epochs 50 \
  --max_vocab 20000 \
  --max_len 64 \
  --emb_dim 256 \
  --hid_dim 512
```

### Training Parameters
- `--data_path`: CSV file with 'prompt,reply' columns
- `--model_dir`: Output directory (default: 'artifacts')
- `--epochs`: Training epochs (default: 20)
- `--batch_size`: Batch size (default: 64)
- `--max_vocab`: Maximum vocabulary size (default: 12000)
- `--max_len`: Maximum sequence length (default: 32)
- `--emb_dim`: Embedding dimension (default: 128)
- `--hid_dim`: Hidden state dimension (default: 256)

## 🖥️ Usage

### 1. Streamlit Web Interface
```bash
streamlit run chatbot_streamlit.py
```
- Interactive chat interface
- Real-time conversation
- Settings sidebar for decoding options
- Chat history management

### 2. Flask REST API
```bash
python chatbot_flask.py
```

**Endpoints:**
- `GET /` - Web interface
- `GET /health` - Health check
- `POST /chat` - Chat with greedy decoding
- `POST /chat/beam` - Chat with beam search

**API Examples:**
```bash
# Basic chat
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'

# Beam search with options
curl -X POST http://127.0.0.1:8000/chat/beam \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "beam_width": 3, "max_tokens": 20}'
```

### 3. Programmatic Usage
```python
from inference_utils import load_artifacts, greedy_decode

# Load model
model, tokenizer, meta = load_artifacts('artifacts')

# Generate response
response = greedy_decode(model, tokenizer, meta, "hello")
print(response)
```

### 4. Interactive Examples
```bash
python run_examples.py
```

## 📊 Data Format

### Training Data (CSV)
```csv
prompt,reply
hi,hello! how can i help you today?
how are you,i'm doing great thanks for asking!
what is ai,artificial intelligence is machine intelligence
bye,goodbye! have a great day!
```

### Tips for Good Training Data
- Use 1000+ conversation pairs for decent performance
- Ensure diverse, natural conversation patterns
- Keep responses concise and relevant
- Include various greeting/farewell patterns
- Add domain-specific conversations for specialized bots

## 🧠 Model Architecture

### Encoder-Decoder with GRU
```
Input Text → Tokenizer → Encoder GRU → Context Vector
                                          ↓
Output Text ← Detokenizer ← Decoder GRU ← Context Vector
```

### Key Components
- **Shared Embedding**: Efficient parameter sharing
- **Encoder GRU**: Processes input sequence
- **Decoder GRU**: Generates response with teacher forcing
- **Attention**: Context-aware response generation

## ⚡ Performance

### Optimization Tips
- **GPU Acceleration**: Install `tensorflow-gpu` for faster training
- **Batch Size**: Increase for faster training (if memory allows)
- **Vocabulary**: Larger vocab for better coverage, smaller for speed
- **Sequence Length**: Longer sequences for detailed responses

### Typical Performance
- **Training Time**: 5-30 minutes (depends on data size and parameters)
- **Inference Speed**: 50-200ms per response
- **Memory Usage**: 200-500MB (loaded model)

## 🔧 Customization

### 1. Model Architecture
```python
# Modify in train_gru_chatbot.py
def build_train_model(vocab_size, emb_dim, hid_dim, max_len):
    # Add LSTM layers, attention, dropout, etc.
    pass
```

### 2. Decoding Strategies
```python
# Add to inference_utils.py
def temperature_sampling(model, tokenizer, meta, prompt, temperature=0.7):
    # Implement temperature-based sampling
    pass
```

### 3. Preprocessing
```python
# Modify cleaning function
def clean_text(s: str) -> str:
    # Add custom preprocessing steps
    pass
```

## 🐛 Troubleshooting

### Common Issues

**1. Model Not Loading**
```
Error: Model directory not found
Solution: Train the model first with `python train_gru_chatbot.py`
```

**2. Poor Response Quality**
```
Issue: Repetitive or nonsensical responses
Solution: 
- Add more diverse training data
- Increase training epochs
- Tune model parameters
```

**3. Memory Errors**
```
Issue: Out of memory during training
Solution:
- Reduce batch_size
- Reduce model dimensions (emb_dim, hid_dim)
- Use smaller vocabulary
```

**4. Slow Performance**
```
Issue: Slow inference
Solution:
- Use greedy decoding instead of beam search
- Reduce max_tokens
- Use GPU acceleration
```

### Debugging Commands
```bash
# Test model loading
python -c "from inference_utils import load_artifacts; load_artifacts('artifacts')"

# Validate training data
python -c "import pandas as pd; print(pd.read_csv('sample_pairs.csv').head())"

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 📈 Advanced Features

### 1. Model Evaluation
```python
# Add BLEU score evaluation
from nltk.translate.bleu_score import sentence_bleu

def evaluate_model(model, test_data):
    # Calculate BLEU scores
    pass
```

### 2. Attention Visualization
```python
# Visualize attention weights
def plot_attention(attention_weights, input_text, output_text):
    # Create attention heatmap
    pass
```

### 3. Fine-tuning
```python
# Fine-tune pre-trained model
def fine_tune(model, new_data, learning_rate=1e-4):
    # Continue training on new data
    pass
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "chatbot_flask.py"]
```

### Cloud Deployment
- **AWS**: Use EC2 + ELB for scalability
- **Google Cloud**: Deploy on Cloud Run
- **Azure**: Use Container Instances
- **Heroku**: Simple deployment with Procfile

### Monitoring
- Add logging for conversation tracking
- Monitor response times and accuracy
- Implement user feedback collection

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- TensorFlow/Keras team for the excellent deep learning framework
- Streamlit for the easy-to-use web interface framework
- Flask for the lightweight REST API framework

## 📞 Support

If you encounter issues or have questions:
1. Check the troubleshooting section
2. Run `python run_examples.py` for diagnostics
3. Create an issue with detailed error messages

---

**Happy Chatbot Building! 🤖✨**