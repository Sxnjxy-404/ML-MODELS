#======================================================
# chatbot_flask.py — Flask REST API
# Run: python chatbot_flask.py (http://127.0.0.1:8000)
#======================================================

from flask import Flask, request, jsonify, render_template_string
import os
import traceback
from inference_utils import load_artifacts, greedy_decode, beam_search_decode

MODEL_DIR = 'artifacts'

app = Flask(__name__)

# Global variables to hold loaded model
model = None
tokenizer = None
meta = None
model_loaded = False
load_error = None

def load_model():
    """Load model on startup"""
    global model, tokenizer, meta, model_loaded, load_error
    try:
        model, tokenizer, meta = load_artifacts(MODEL_DIR)
        model_loaded = True
        print("Model loaded successfully!")
        return True
    except Exception as e:
        load_error = str(e)
        print(f"Error loading model: {e}")
        return False

# Load model when starting the app
load_model()

@app.route('/')
def home():
    """Simple web interface for testing"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GRU Chatbot API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .chat-box { border: 1px solid #ddd; padding: 20px; margin: 20px 0; height: 300px; overflow-y: scroll; }
            .input-group { display: flex; margin: 10px 0; }
            input[type="text"] { flex: 1; padding: 10px; border: 1px solid #ddd; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
            .user { background: #e3f2fd; text-align: right; }
            .bot { background: #f5f5f5; }
            .error { color: red; background: #ffebee; padding: 10px; border-radius: 5px; }
            .info { color: #666; background: #f0f0f0; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 GRU Chatbot API</h1>
            
            {% if not model_loaded %}
            <div class="error">
                <strong>Model not loaded:</strong> {{ load_error }}<br>
                Please run: <code>python train_gru_chatbot.py</code> first.
            </div>
            {% else %}
            <div class="info">
                <strong>Model Status:</strong> Loaded ✅<br>
                <strong>Vocabulary Size:</strong> {{ meta.vocab_size }}<br>
                <strong>Max Length:</strong> {{ meta.max_len }}
            </div>
            
            <div id="chat-box" class="chat-box"></div>
            
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Send</button>
            </div>
            
            <h3>API Endpoints:</h3>
            <ul>
                <li><strong>GET /health</strong> - Health check</li>
                <li><strong>POST /chat</strong> - Send message (JSON: {"message": "your message"})</li>
                <li><strong>POST /chat/beam</strong> - Send message with beam search</li>
            </ul>
            {% endif %}
        </div>
        
        <script>
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            function addMessage(message, isUser) {
                const chatBox = document.getElementById('chat-box');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + (isUser ? 'user' : 'bot');
                messageDiv.textContent = (isUser ? 'You: ' : 'Bot: ') + message;
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }
            
            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;
                
                addMessage(message, true);
                input.value = '';
                
                fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.reply) {
                        addMessage(data.reply, false);
                    } else if (data.error) {
                        addMessage('Error: ' + data.error, false);
                    }
                })
                .catch(error => {
                    addMessage('Error: ' + error, false);
                });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html, 
                                model_loaded=model_loaded, 
                                load_error=load_error,
                                meta=meta if model_loaded else None)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "error": load_error if not model_loaded else None
    }
    
    if model_loaded and meta:
        status["model_info"] = {
            "vocab_size": meta.get("vocab_size"),
            "max_len": meta.get("max_len"),
            "emb_dim": meta.get("emb_dim"),
            "hid_dim": meta.get("hid_dim")
        }
    
    return jsonify(status)

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint using greedy decoding"""
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "details": load_error
        }), 500
    
    try:
        data = request.get_json(force=True)
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Empty message"}), 400
        
        # Get optional parameters
        max_tokens = data.get('max_tokens', 30)
        max_tokens = min(max(max_tokens, 5), 100)  # Clamp between 5-100
        
        # Generate reply
        reply = greedy_decode(model, tokenizer, meta, message, max_tokens)
        
        return jsonify({
            "reply": reply,
            "method": "greedy",
            "input_message": message
        })
        
    except Exception as e:
        return jsonify({
            "error": "Generation failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/chat/beam', methods=['POST'])
def chat_beam():
    """Chat endpoint using beam search decoding"""
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "details": load_error
        }), 500
    
    try:
        data = request.get_json(force=True)
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Empty message"}), 400
        
        # Get optional parameters
        max_tokens = data.get('max_tokens', 30)
        beam_width = data.get('beam_width', 3)
        
        # Validate parameters
        max_tokens = min(max(max_tokens, 5), 100)
        beam_width = min(max(beam_width, 1), 5)
        
        # Generate reply
        reply = beam_search_decode(model, tokenizer, meta, message, beam_width, max_tokens)
        
        return jsonify({
            "reply": reply,
            "method": "beam_search",
            "beam_width": beam_width,
            "input_message": message
        })
        
    except Exception as e:
        return jsonify({
            "error": "Generation failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/reload', methods=['POST'])
def reload_model():
    """Reload the model (useful after retraining)"""
    global model_loaded
    success = load_model()
    
    return jsonify({
        "success": success,
        "model_loaded": model_loaded,
        "error": load_error if not success else None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting GRU Chatbot Flask API...")
    print(f"Model loaded: {model_loaded}")
    if model_loaded:
        print(f"Vocabulary size: {meta.get('vocab_size', 'N/A')}")
        print(f"Max sequence length: {meta.get('max_len', 'N/A')}")
    else:
        print(f"Load error: {load_error}")
    
    print("\nAvailable endpoints:")
    print("  GET  /          - Web interface")
    print("  GET  /health    - Health check")
    print("  POST /chat      - Chat with greedy decoding")
    print("  POST /chat/beam - Chat with beam search")
    print("  POST /reload    - Reload model")
    print("\nStarting server on http://127.0.0.1:8000")
    
    app.run(host='0.0.0.0', port=8000, debug=False)