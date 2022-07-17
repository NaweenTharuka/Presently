import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle

################################## Variables & Paramaters #####################################
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

POSE_MODEL_PATH = "users/models/body_language.pkl"

################################ POSE ESTIMATION MODEL & VIDEO ################################

def req_objects(VIDEO_FILE):
    with open(POSE_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(VIDEO_FILE)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps =  int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    POSE_VIDEO_FILE = os.path.split(VIDEO_FILE)[1]
    POSE_VIDEO_PATH = 'users/static/users/pose_detected/{}'.format(POSE_VIDEO_FILE)
    video_pose = cv2.VideoWriter(POSE_VIDEO_PATH, fourcc, fps, (width, height))

    return model, cap, video_pose

###############################################################################################
def init_holistic_model(model, cap, video_pose):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is not None:

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                
                # Make Detections
                results = holistic.process(image)
                # print(results.face_landmarks)
                
                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # 1. Draw face landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                        )
                
                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    
                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                    
                    # Concate rows
                    row = pose_row+face_row

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    # print(body_language_class, body_language_prob)
                    
                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    
                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    video_pose.write(image)

                except:
                    pass    

            if not ret:
                break

        cap.release()
        cv2.destroyAllWindows()

def body_language_decoder(VIDEO_FILE):
    model, cap, video_pose = req_objects(VIDEO_FILE)
    init_holistic_model(model, cap, video_pose)