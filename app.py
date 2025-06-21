import os
import datetime
import logging
import io
import base64
import uuid

import cv2
import pandas as pd
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from deepface import DeepFace

from flask import Flask, request, jsonify, render_template

# --- App & Logger Setup ---
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- Constants & Directory Setup ---
LOG_FILE = "wellbeing_logs.csv"
CAPTURED_IMAGE_DIR = "captured_images"
TEMP_AUDIO_DIR = "temp_audio"

os.makedirs(CAPTURED_IMAGE_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# --- Caching the Model ---
voice_model = None
voice_feature_extractor = None

def load_voice_emotion_model():
    global voice_model, voice_feature_extractor
    if voice_model is None:
        logging.info("Loading voice emotion model for the first time...")
        model_name = "superb/wav2vec2-base-superb-er"
        voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        voice_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        logging.info("Voice emotion model loaded.")
    return voice_model, voice_feature_extractor

# --- Analysis Functions ---
def analyze_voice_emotion(audio_file_path):
    try:
        model, feature_extractor = load_voice_emotion_model()
        y, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        if y.shape[0] == 0:
            logging.warning(f"Audio file {audio_file_path} was empty.")
            return "Error: Invalid or empty audio"
        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        return model.config.id2label[predicted_id]
    except Exception as e:
        logging.exception(f"Voice emotion analysis failed for file {audio_file_path}: {e}")
        return "Error: Voice analysis failed"

def analyze_emotion_from_data(image_bytes, detector_backend="retinaface"):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            return "Error: Could not decode image"
        
        # Use a fallback detector if the selected one fails
        try:
            result = DeepFace.analyze(
                img_path=img_np, actions=['emotion'],
                detector_backend=detector_backend, enforce_detection=False
            )
        except Exception as detector_error:
            logging.warning(f"Detector '{detector_backend}' failed: {detector_error}. Falling back to 'opencv'.")
            result = DeepFace.analyze(
                img_path=img_np, actions=['emotion'],
                detector_backend='opencv', enforce_detection=False
            )

        if isinstance(result, list) and len(result) > 0:
            return result[0].get("dominant_emotion", "No face detected")
        else:
            return "No face detected"
    except Exception as e:
        logging.exception(f"Face emotion analysis failed with backend {detector_backend}: {e}")
        return "Error: Face analysis failed"

def assess_stress_enhanced(face_emotion, sleep_hours, activity_level, voice_emotion):
    activity_map = {"Very Low": 3, "Low": 2, "Moderate": 1, "High": 0}
    emotion_map = { "angry": 2, "disgust": 2, "fear": 2, "sad": 2, "neutral": 1, "surprise": 1, "happy": 0 }
    face_emotion_score = emotion_map.get(str(face_emotion).lower(), 1)
    voice_emotion_score = emotion_map.get(str(voice_emotion).lower(), 1)
    emotion_score = round((face_emotion_score + voice_emotion_score) / 2) if voice_emotion != "N/A" else face_emotion_score
    activity_score = activity_map.get(str(activity_level), 1)
    try:
        sleep_hours = float(sleep_hours)
        sleep_score = 0 if sleep_hours >= 7 else (1 if sleep_hours >= 5 else 2)
    except (ValueError, TypeError):
        sleep_score, sleep_hours = 2, 0
    stress_score = emotion_score + activity_score + sleep_score
    feedback = f"**Your potential stress score is {stress_score} (lower is better).**\n\n**Breakdown:**\n"
    feedback += f"- Face Emotion: {face_emotion} (score: {face_emotion_score})\n"
    feedback += f"- Voice Emotion: {voice_emotion} (score: {voice_emotion_score})\n"
    feedback += f"- Sleep: {sleep_hours} hours (score: {sleep_score})\n"
    feedback += f"- Activity: {activity_level} (score: {activity_score})\n"
    if stress_score <= 2:
        feedback += "\nGreat job! You seem to be in a good space."
    elif stress_score <= 4:
        feedback += "\nYou're doing okay, but remember to be mindful of your rest and mood."
    else:
        feedback += "\nConsider taking some time for self-care. Improving sleep or gentle activity might help."
    return feedback, stress_score

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_face', methods=['POST'])
def analyze_face_endpoint():
    data = request.json
    detector = data.get('detector', 'retinaface')
    image_data = base64.b64decode(data['image'].split(',')[1])
    emotion = analyze_emotion_from_data(image_data, detector_backend=detector)
    image_path = "N/A"
    if not emotion.startswith("Error:") and not emotion == "No face detected":
        filename = f"face_{uuid.uuid4()}.jpg"
        image_path = os.path.join(CAPTURED_IMAGE_DIR, filename)
        with open(image_path, "wb") as f:
            f.write(image_data)
    return jsonify({'emotion': emotion, 'image_path': image_path})

@app.route('/analyze_voice', methods=['POST'])
def analyze_voice_endpoint():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({'error': 'No audio file provided'}), 400
    temp_filename = f"{uuid.uuid4()}.webm"
    temp_filepath = os.path.join(TEMP_AUDIO_DIR, temp_filename)
    try:
        audio_file.save(temp_filepath)
        emotion = analyze_voice_emotion(temp_filepath)
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
    return jsonify({'voice_emotion': emotion})

@app.route('/log_checkin', methods=['POST'])
def log_checkin_endpoint():
    data = request.json
    feedback, stress_score = assess_stress_enhanced(
        data['emotion'], data['sleep_hours'], data['activity_level'], data['voice_emotion']
    )
    # *** FIX: Format timestamp as a consistent string BEFORE saving ***
    new_log_entry = {
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "face_emotion": data['emotion'],
        "voice_emotion": data.get('voice_emotion', 'N/A'),
        "sleep_hours": data['sleep_hours'],
        "activity_level": data['activity_level'],
        "stress_score": stress_score,
        "detector_backend": data.get('detector', 'retinaface'),
        "image_path": data.get('image_path', 'N/A')
    }
    try:
        header = not os.path.exists(LOG_FILE)
        df_new = pd.DataFrame([new_log_entry])
        df_new.to_csv(LOG_FILE, mode='a', header=header, index=False)
        return jsonify({'feedback': feedback, 'stress_score': stress_score, 'status': 'success'})
    except Exception as e:
        logging.exception(f"Could not save log: {e}")
        return jsonify({'error': f'Could not save log: {e}'}), 500

@app.route('/get_logs', methods=['GET'])
def get_logs_endpoint():
    if not os.path.exists(LOG_FILE):
        return jsonify({'data': [], 'columns': []})
    try:
        df = pd.read_csv(LOG_FILE)
        # *** FIX: No need to parse/reformat timestamps. They are already correct strings. ***
        return jsonify({
            'data': df.to_dict(orient='records'),
            'columns': df.columns.tolist()
        })
    except pd.errors.EmptyDataError:
        return jsonify({'data': [], 'columns': []})
    except Exception as e:
        logging.exception(f"Could not read logs: {e}")
        return jsonify({'error': 'Could not read logs'}), 500

@app.route('/clear_logs', methods=['POST'])
def clear_logs_endpoint():
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        for directory in [CAPTURED_IMAGE_DIR, TEMP_AUDIO_DIR]:
            if os.path.exists(directory):
                for f in os.listdir(directory):
                    os.remove(os.path.join(directory, f))
        return jsonify({'status': 'success', 'message': 'All logs and images cleared.'})
    except Exception as e:
        logging.exception(f"Error clearing logs: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    load_voice_emotion_model()
    app.run(debug=True, host='0.0.0.0')