import os
from json_tricks import load
import numpy as np
import librosa
from pydub import AudioSegment, effects
import noisereduce as nr
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pyaudio

from pydub import AudioSegment

################################ Variables & Paramaters #####################################
saved_model_path = "users/models/emotion_detection_audio.json"
saved_weights_path = "users/models/emotion_detection_audio.h5"

emotions = {
            0 : 'neutral',
            1 : 'calm',
            2 : 'happy',
            3 : 'sad',
            4 : 'angry',
            5 : 'fearful',
            6 : 'disgust',
            7 : 'suprised'   
            }

################################ Load Emotion Detection #####################################
with open(saved_model_path, 'r') as json_file:
    json_savedModel = json_file.read()
    
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(saved_weights_path)
model.compile(
            loss='categorical_crossentropy', 
            optimizer='RMSProp', 
            metrics=['categorical_accuracy']
            )

emo_list = list(emotions.values())

################################ Utility Functions ##########################################
def preprocess(file_path, frame_length = 2048, hop_length = 512):
    _, sr = librosa.load(path = file_path, sr = None)
    rawsound = AudioSegment.from_file(file_path, duration = None) 
    normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32')
    final_x = nr.reduce_noise(normal_x, sr=sr)
        
        
    f1 = librosa.feature.rms(final_x, frame_length=frame_length, hop_length=hop_length, center=True, pad_mode='reflect').T 
    f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frame_length, hop_length=hop_length,center=True).T 
    f3 = librosa.feature.mfcc(final_x, sr=sr, S=None, n_mfcc=13, hop_length = hop_length).T

    X = np.concatenate((f1, f2, f3), axis = 1)
    
    X_3D = np.expand_dims(X, axis=0)
    return X_3D

def predictions(X_3D):
    predictions = model.predict(X_3D).squeeze()
    likelihood = predictions * 100 
    return likelihood

def detect_emotions_audio(file_path):
    X_3D = preprocess(file_path)
    likelihood = predictions(X_3D)

    emotion_probs = {}
    for prob, emotion in zip(likelihood, emo_list):
        emotion_probs[emotion] = round(float(prob), 3)
    return emo_list[likelihood.argmax()], emotion_probs
    