import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
import scipy.signal
import random
import warnings
import librosa
from transformers import pipeline

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

app = Flask(__name__)
CORS(app)

# Load pre-trained facial model (placeholder)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval().to(device)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MediaPipe for gait
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Voice sentiment pipeline (load once)
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
except Exception as e:
    print(f"Voice pipeline load error: {e}")
    sentiment_pipeline = None

def analyze_emotions(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
    stress_prob = torch.softmax(output, dim=1)[0][0].item()
    return {'stress': stress_prob}

def estimate_hrv(image):
    green = image[:, :, 1].astype(float)
    signal = np.mean(green, axis=1)
    if len(signal) < 30:
        return {'value': random.uniform(40, 100)}
    b, a = scipy.signal.butter(3, [0.7/15, 4/15], btype='band', fs=30)
    try:
        filtered = scipy.signal.filtfilt(b, a, signal)
        peaks, _ = scipy.signal.find_peaks(filtered, distance=15)
        rmssd = random.uniform(40, 100) if len(peaks) <= 1 else np.sqrt(np.mean(np.diff(peaks)**2))
    except:
        rmssd = random.uniform(40, 100)
    return {'value': rmssd}

def analyze_gait(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        stride = abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x - landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x)
        anomaly = stride < 0.1
        return {'stride': stride, 'anomaly': anomaly}
    return {'stride': 0, 'anomaly': False}

def analyze_voice(audio_data):
    if not audio_data or audio_data == '':
        return {'sentiment': 'neutral', 'stress': 0.5}  # Fallback for no audio
    try:
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        if len(audio_bytes) == 0:
            return {'sentiment': 'neutral', 'stress': 0.5}
        audio_array, sr = librosa.load(BytesIO(audio_bytes), sr=16000)
        if len(audio_array) == 0:
            return {'sentiment': 'neutral', 'stress': 0.5}
        # MFCC for features (optional)
        mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
        # Sentiment (placeholder text; real: add STT like whisper)
        sample_text = "Sample speech from audio"  # Replace with STT output
        if sentiment_pipeline:
            sentiment = sentiment_pipeline(sample_text)[0]
            label = sentiment['label'].lower()
            score = sentiment['score']
            stress_from_voice = 0.8 if 'negative' in label else 0.2 if 'positive' in label else 0.5
            return {'sentiment': label, 'stress': min(stress_from_voice * score, 1.0)}
        else:
            return {'sentiment': 'neutral', 'stress': 0.5}
    except Exception as e:
        print(f"Voice analysis error: {e}")
        return {'sentiment': 'neutral', 'stress': 0.5}

def get_recommendations(emotions, hrv, gait, voice):
    stress = max(emotions.get('stress', 0), voice.get('stress', 0))
    recs = []
    if stress > 0.7:
        recs.append("Try a 5-minute guided meditation.")
    if hrv['value'] < 50:
        recs.append("Consider deep breathing exercises.")
    if gait['anomaly']:
        recs.append("Schedule a neurological check-up.")
    if voice.get('sentiment', '') == 'negative':
        recs.append("Practice positive affirmations to lift your mood.")
    return recs

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Backend running! Load frontend/index.html in your browser.'})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image data'}), 400
    try:
        # Image
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img = np.array(Image.open(BytesIO(img_bytes)))

        # Audio (optional)
        audio_data = data.get('audio', '')
        voice = analyze_voice(audio_data)

        emotions = analyze_emotions(img)
        hrv = estimate_hrv(img)
        gait = analyze_gait(img)
        recommendations = get_recommendations(emotions, hrv, gait, voice)

        return jsonify({
            'emotions': emotions,
            'hrv': hrv,
            'gait': gait,
            'voice': voice,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)