from fer import Video
from fer import FER
import os
import sys
import pandas as pd
import numpy as np

import cv2
from tensorflow.keras.models import load_model
                
################################ Variables & Paramaters #####################################
CASC_PATH = "users/models/harr_feature.xml"
MODEL_PATH = "users/models/emotion_detection_video.h5"
  
############################# Load Emotion Detection VIDEO ##################################

def _get_labels():
  return {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral",
        }

def tosquare(bbox):
        x, y, w, h = bbox
        if h > w:
            diff = h - w
            x -= diff // 2
            w += diff
        elif w > h:
            diff = w - h
            y -= diff // 2
            h += diff
        if w != h:
            print(f"{w} is not {h}")

        return (x, y, w, h)

def __apply_offsets(face_coordinates):
    x, y, width, height = face_coordinates
    x_off, y_off = (10, 10)
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def __preprocess_input(x, v2=False):
        x = x.astype("float32")
        x = x / 255.0
        if v2:
            x = x - 0.5
            x = x * 2.0
        return x

def pad(image):
        PADDING = 40
        row, col = image.shape[:2]
        bottom = image[row - 2 : row, 0:col]
        mean = cv2.mean(bottom)[0]

        padded_image = cv2.copyMakeBorder(
            image,
            top = PADDING,
            bottom = PADDING,
            left = PADDING,
            right= PADDING,
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean],
        )
        return padded_image


def detect_emotions_video(location_videofile, NumberofFrames):
    PADDING = 40
    emotion_labels = _get_labels()
    arry = {}

    vidcap = cv2.VideoCapture(location_videofile)

    success,image = vidcap.read()
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    faceCascade = cv2.CascadeClassifier(CASC_PATH)

    while vidcap.isOpened():
        score = 0
        success,image = vidcap.read()

        if success:
            if frame_count > NumberofFrames+1:
              count += frame_count/(NumberofFrames+1) 
            else:
              count += 1
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
            gray_image_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
            gray_image_array,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

            if len(faces) == 1:
              gray_img = pad(gray_image_array)

              emotions = []
              for face_coordinates in faces:
                  face_coordinates = tosquare(face_coordinates)
                  x1, x2, y1, y2 = __apply_offsets(face_coordinates)
                  
                  x1 += PADDING
                  x2 += PADDING
                  y1 += PADDING
                  y2 += PADDING            
                  x1 = np.clip(x1, a_min=0, a_max=None)
                  y1 = np.clip(y1, a_min=0, a_max=None)

                  gray_face = gray_img[max(0, y1 - PADDING):y2 + PADDING,
                                      max(0, x1 - PADDING):x2 + PADDING]
                  gray_face = gray_img[y1:y2, x1:x2]

                  model = load_model(MODEL_PATH, compile=True)
                  model.make_predict_function()

                  try:
                    gray_face = cv2.resize(gray_face, model.input_shape[1:3])
                  except Exception as e:
                    print("Cannot resize "+str(e))
                    continue
                  
                  gray_face = __preprocess_input(gray_face, True)
                  gray_face = np.expand_dims(np.expand_dims(gray_face, 0), -1)

                  emotion_prediction = model.predict(gray_face)[0]
                  labelled_emotions = {
                      emotion_labels[idx]: round(float(score), 2)
                      for idx, score in enumerate(emotion_prediction)
                  }

                  emotions.append(
                      dict(box=face_coordinates, emotions=labelled_emotions)
                  )
              top_emotions  = [max(e["emotions"], key=lambda key: e["emotions"][key]) for e in emotions]
              if len(top_emotions):
                for top_emotion in emotions[0]["emotions"]:
                  if top_emotion in arry.keys():
                    arry.update({top_emotion: arry[top_emotion] + emotions[0]["emotions"][top_emotion]})
                  else:
                    arry[top_emotion] = score

        else:
            vidcap.release()
            break
    if len(arry) == 0:
      return "Cannot detect emotion", arry
    else:
      return max(arry, key=arry.get), arry