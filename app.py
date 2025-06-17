import os
import datetime
import logging
import io
import base64

import cv2
import pandas as pd
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import soundfile as sf
from deepface import DeepFace

from flask import Flask, request, jsonify, render_template

# --- App & Logger Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Constants ---
LOG_FILE = "wellbeing_logs.csv"
CAPTURED_IMAGE_DIR = "captured_images"
os.makedirs(CAPTURED_IMAGE_DIR, exist_ok=True)

# --- Caching the Model (Simple in-memory cache) ---
voice_model = None
voice_feature_extractor = None

def load_voice_emotion_model():
    """Loads the voice emotion model into global variables."""
    global voice_model, voice_feature_extractor
    if voice_model is None or voice_feature_extractor is None:
        logging.info("Loading voice emotion model for the first time...")
        model_name = "superb/wav2vec2-base-superb-er"
        voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        voice_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        logging.info("Voice emotion model loaded.")
    return voice_model, voice_feature_extractor

# --- Analysis Functions ---

def analyze_voice_emotion(audio_bytes):
    try:
        model, feature_extractor = load_voice_emotion_model()
        y, sr = sf.read(io.BytesIO(audio_bytes))
        if len(y.shape) > 1:  # If stereo, convert to mono
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        emotion_labels = model.config.id2label
        emotion = emotion_labels[predicted_id]
        return emotion
    except Exception as e:
        logging.error(f"Voice emotion analysis failed: {e}")
        return "Error: Voice analysis failed"

def analyze_emotion_from_data(image_bytes, detector_backend="mediapipe"):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            return "Error: Could not decode image"
        result = DeepFace.analyze(
            img_path=img_np,
            actions=['emotion'],
            detector_backend=detector_backend,
            enforce_detection=False
        )
        if isinstance(result, list):
            return result[0].get("dominant_emotion", "No face detected")
        else:
            return result.get("dominant_emotion", "No face detected")
    except Exception as e:
        logging.error(f"Face emotion analysis failed: {e}")
        return f"Error: {str(e)}"

def assess_stress_enhanced(emotion, sleep_hours, activity_level):
    activity_map = {"Very Low": 3, "Low": 2, "Moderate": 1, "High": 0}
    emotion_map = {
        "angry": 2, "disgust": 2, "fear": 2, "sad": 2,
        "neutral": 1, "surprise": 1, "happy": 0
    }
    emotion_score = emotion_map.get(str(emotion).lower(), 1)
    activity_score = activity_map.get(str(activity_level), 1)
    try:
        sleep_hours = float(sleep_hours)
        sleep_score = 0 if sleep_hours >= 7 else (1 if sleep_hours >= 5 else 2)
    except (ValueError, TypeError):
        sleep_score = 2
        sleep_hours = 0
    stress_score = emotion_score + activity_score + sleep_score
    feedback = f"**Your potential stress score is {stress_score} (lower is better).**\n\n**Breakdown:**\n"
    feedback += f"- Emotion: {emotion} (score {emotion_score})\n"
    feedback += f"- Sleep: {sleep_hours} hours (score {sleep_score})\n"
    feedback += f"- Activity: {activity_level} (score {activity_score})\n"
    if stress_score <= 1:
        feedback += "\nGreat job! You seem well-rested and positive."
    elif stress_score <= 3:
        feedback += "\nYou're doing okay, but keep an eye on your rest and mood."
    else:
        feedback += "\nConsider improving your sleep, activity, or talking to someone if you feel stressed."
    return feedback, stress_score

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_face', methods=['POST'])
def analyze_face_endpoint():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    detector = data.get('detector', 'mediapipe')
    image_data = base64.b64decode(data['image'].split(',')[1])
    emotion = analyze_emotion_from_data(image_data, detector_backend=detector)
    image_path = "N/A"
    if not emotion.startswith("Error:") and not emotion == "No face detected":
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"face_{timestamp_str}.jpg"
        image_path = os.path.join(CAPTURED_IMAGE_DIR, filename)
        with open(image_path, "wb") as f:
            f.write(image_data)
    return jsonify({'emotion': emotion, 'image_path': image_path})

@app.route('/analyze_voice', methods=['POST'])
def analyze_voice_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    emotion = analyze_voice_emotion(audio_bytes)
    return jsonify({'voice_emotion': emotion})

@app.route('/log_checkin', methods=['POST'])
def log_checkin_endpoint():
    data = request.json
    required_fields = ['emotion', 'sleep_hours', 'activity_level', 'image_path', 'voice_emotion']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing data for logging'}), 400
    feedback, stress_score = assess_stress_enhanced(
        data['emotion'], data['sleep_hours'], data['activity_level']
    )
    new_log_entry = {
        "timestamp": datetime.datetime.now(),
        "emotion": data['emotion'],
        "voice_emotion": data.get('voice_emotion', 'N/A'),
        "sleep_hours": data['sleep_hours'],
        "activity_level": data['activity_level'],
        "stress_score": stress_score,
        "feedback_summary": feedback.split('\n\n**Breakdown:**')[0],
        "detector_backend": data.get('detector', 'mediapipe'),
        "image_path": data.get('image_path', 'N/A')
    }
    try:
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
        else:
            df = pd.DataFrame(columns=new_log_entry.keys())
        new_df_entry = pd.DataFrame([new_log_entry])
        df = pd.concat([df, new_df_entry], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
        return jsonify({'feedback': feedback, 'stress_score': stress_score, 'status': 'success'})
    except Exception as e:
        logging.error(f"Could not save log: {e}")
        return jsonify({'error': f'Could not save log: {e}'}), 500

@app.route('/get_logs', methods=['GET'])
def get_logs_endpoint():
    if not os.path.exists(LOG_FILE):
        return jsonify({'data': [], 'columns': []})
    df = pd.read_csv(LOG_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    return jsonify({
        'data': df.to_dict(orient='records'),
        'columns': df.columns.tolist()
    })

@app.route('/clear_logs', methods=['POST'])
def clear_logs_endpoint():
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
            for f in os.listdir(CAPTURED_IMAGE_DIR):
                os.remove(os.path.join(CAPTURED_IMAGE_DIR, f))
            return jsonify({'status': 'success', 'message': 'All logs and images cleared.'})
        except Exception as e:
            logging.error(f"Error clearing logs: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'success', 'message': 'No log file to clear.'})

if __name__ == '__main__':
    load_voice_emotion_model()
    app.run(debug=True)
    # For production, use a WSGI server like Gunicorn:
    # gunicorn --workers 4 --bind 0.0.0.0:5000 app:app