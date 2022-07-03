import speech_recognition as sr
import moviepy.editor as mp

clip = mp.VideoFileClip(r"F:\\CDAP-PRESENTLY\\21_22-j-02\\Presently\\presently\\media\\video\\22\\publicspeech.mp4")
clip.audio.write_audiofile(r"converted_mp3.wav")
r = sr.Recognizer()
audio =sr.AudioFile(r"converted_mp3.wav")
with audio as source:
    r.pause_threshold = 1
    r.adjust_for_ambient_noise(source, duration=1)
    audio_file = r.record(source)  
result = r.recognize_google(audio_file)
print(result)