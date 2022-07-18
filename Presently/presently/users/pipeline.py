import os
import warnings
warnings.filterwarnings("ignore")

import moviepy.editor as mp
import speech_recognition as sr

import users.myprosody as myprosody
from users.english_checker import EnglishChecker
from users.emotiondetectionvideo import detect_emotions_video
from users.emotiondetectionaudio import detect_emotions_audio
from users.correct_body_language_decoder import body_language_decoder

def audio_execution(video_path, audio_path):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    r = sr.Recognizer()
    audio =sr.AudioFile(audio_path)
    with audio as source:
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        audio_file = r.record(source)  
    r.recognize_google(audio_file)

def V2A(video_path, prosody_path = "users/myprosody/dataset/essen/audios"):
    audio_file = os.path.split(video_path)[1].split('.')[0] + '.wav'
    audio_path = 'users/audios/{}'.format(audio_file)
    audio_path_prosody = '{}/{}'.format(prosody_path, audio_file)

    audio_execution(video_path, audio_path)
    audio_execution(video_path, audio_path_prosody)

    print('Audio File Generated at: {}'.format(audio_path))
    print('Audio File Generated at: {}'.format(audio_path_prosody))

    assert os.path.exists(audio_path), "Audio File not found in General path"
    assert os.path.exists(audio_path_prosody), "Audio File not found in Prosody path"
    return audio_path

def prosody_execution(audio_path, c="users/myprosody"):
    p = os.path.split(audio_path)[1].split('.')[0]

    try:
        myspsyl= myprosody.myspsyl(p,c)
        mysppaus= myprosody.mysppaus(p,c)
        myspsr= myprosody.myspsr(p,c)
        myspatc= myprosody.myspatc(p,c)
        myspst= myprosody.myspst(p,c)
        myspod= myprosody.myspod(p,c)
        myspbala= myprosody.myspbala(p,c)
        myspf0mean= myprosody.myspf0mean(p,c)
        myspf0sd= myprosody.myspf0sd(p,c)
        myspf0med= myprosody.myspf0med(p,c)
        myspf0min= myprosody.myspf0min(p,c)
        myspf0max= myprosody.myspf0max(p,c)
        myspf0q25= myprosody.myspf0q25(p,c)
        myspf0q75= myprosody.myspf0q75(p,c)
        myspgend= myprosody.myspgend(p,c)
        mysppron= myprosody.mysppron(p,c)
        prosody= myprosody.myprosody(p,c)
        
        prosody_context = {
                        'myspsyl': myspsyl,
                        'mysppaus': mysppaus,
                        'myspsr': myspsr,
                        'myspatc': myspatc,
                        'myspst': myspst,
                        'myspod': myspod,
                        'myspbala': myspbala,
                        'myspf0mean': myspf0mean,
                        'myspf0sd': myspf0sd,
                        'myspf0med': myspf0med,
                        'myspf0min': myspf0min,
                        'myspf0max': myspf0max,
                        'myspf0q25': myspf0q25,
                        'myspf0q75': myspf0q75,
                        'myspgend': myspgend,
                        'mysppron': mysppron,
                        'prosody': prosody
                    }
    except:
        prosody_context = {"status": "Try again the sound of the audio was not clear"}
    return prosody_context

def run(video_path):
    audio_path = V2A(video_path) 
    english_checker = EnglishChecker()
    transcription, grammer_corrected = english_checker.checker(audio_path)

    _, emotion_logits_video = detect_emotions_video(video_path, 50)
    body_language_decoder(video_path)

    _, emotion_logits_audio = detect_emotions_audio(audio_path)
    prosody_context = prosody_execution(audio_path)

    return transcription, emotion_logits_video, emotion_logits_audio, prosody_context, grammer_corrected

transcription, emotion_logits_video, emotion_logits_audio, prosody_context, grammer_corrected = run('users/videos/v1.mp4')