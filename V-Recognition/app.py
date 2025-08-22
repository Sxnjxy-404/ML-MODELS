from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import pickle
import pandas as pd
import os
from datetime import datetime
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and encoders with error handling
try:
    model = pickle.load(open('model.pkl', 'rb'))
    label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
    logger.info("✅ Model and encoders loaded successfully")
except FileNotFoundError as e:
    logger.error(f"❌ Model files not found: {e}")
    model = None
    label_encoders = None

# Valid values for each feature
VALID_VALUES = {
    'Outlook': ['Sunny', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Mild', 'Cool'],
    'Humidity': ['High', 'Normal'],
    'Wind': ['Weak', 'Strong']
}

@app.route('/')
def index():
    return render_template('index.html', valid_values=VALID_VALUES)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoders is None:
        return render_template('result.html', 
                             prediction="❌ Model not loaded. Please run train.py first.",
                             confidence=None)
    
    recognizer = sr.Recognizer()
    try:
        # Get available microphones
        mics = sr.Microphone.list_microphone_names()
        if not mics:
            return render_template('result.html', 
                                 prediction="❌ No microphones detected.",
                                 confidence=None)
        
        # Try different microphone indices
        for mic_index in [0, 1, None]:  # None uses default
            try:
                with sr.Microphone(device_index=mic_index) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    logger.info(f"🎙️ Using microphone index: {mic_index}")
                    logger.info("🎙️ Say 4 words: Outlook Temperature Humidity Wind")
                    logger.info("📝 Example: 'Sunny Hot High Weak'")
                    
                    audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
                    voice_input = recognizer.recognize_google(audio)
                    logger.info(f"✅ Recognized: {voice_input}")
                    
                    # Process the voice input
                    result, confidence = process_input(voice_input)
                    return render_template('result.html', 
                                         prediction=result, 
                                         confidence=confidence,
                                         input_text=voice_input)
                    
            except sr.WaitTimeoutError:
                continue  # Try next microphone
            except Exception as e:
                logger.warning(f"Microphone {mic_index} failed: {e}")
                continue
        
        return render_template('result.html', 
                             prediction="❌ Could not access any microphone. Please check your audio settings.",
                             confidence=None)

    except sr.UnknownValueError:
        return render_template('result.html', 
                             prediction="❌ Could not understand audio. Please speak clearly.",
                             confidence=None)
    except sr.RequestError as e:
        return render_template('result.html', 
                             prediction=f"❌ Speech recognition service error: {str(e)}",
                             confidence=None)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('result.html', 
                             prediction=f"❌ Unexpected error: {str(e)}",
                             confidence=None)

@app.route('/predict_text', methods=['POST'])
def predict_text():
    if model is None or label_encoders is None:
        return jsonify({'error': 'Model not loaded. Please run train.py first.'})
    
    try:
        text_input = request.form.get('text_input', '').strip()
        if not text_input:
            return jsonify({'error': 'Please provide input text.'})
        
        result, confidence = process_input(text_input)
        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'input_text': text_input
        })
    except Exception as e:
        return jsonify({'error': f'Error processing text: {str(e)}'})

def process_input(input_text):
    """Process input text and make prediction"""
    features = input_text.strip().title().split()
    
    if len(features) != 4:
        return f"❌ Please provide exactly 4 words: Outlook Temperature Humidity Wind (got {len(features)} words)", None
    
    # Validate input values
    feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    for i, (feature_name, value) in enumerate(zip(feature_names, features)):
        if value not in VALID_VALUES[feature_name]:
            valid_options = ', '.join(VALID_VALUES[feature_name])
            return f"❌ Invalid {feature_name}: '{value}'. Valid options: {valid_options}", None
    
    # Create input dataframe
    input_df = pd.DataFrame([{
        'Outlook': features[0],
        'Temperature': features[1],
        'Humidity': features[2],
        'Wind': features[3]
    }])
    
    logger.info(f"📊 Input DataFrame:\n{input_df}")
    
    # Encode input
    try:
        for col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get prediction probability for confidence
        try:
            probabilities = model.predict_proba(input_df)[0]
            confidence = max(probabilities) * 100
        except:
            confidence = None
        
        result = "✔️ Play Tennis: Yes" if prediction == 1 else "❌ Play Tennis: No"
        
        # Log prediction
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] Prediction: {result} | Input: {' '.join(features)}")
        
        return result, confidence
        
    except Exception as e:
        return f"❌ Error processing input: {str(e)}", None

@app.route('/api/valid_values')
def get_valid_values():
    """API endpoint to get valid values for each feature"""
    return jsonify(VALID_VALUES)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoders_loaded': label_encoders is not None,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)