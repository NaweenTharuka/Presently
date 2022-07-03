from happytransformer import HappyTextClassification,TTSettings, HappyTextToText
import speech_recognition as sr
import moviepy.editor as mp


def convertText():
    clip = mp.VideoFileClip(r"F:\\CDAP-PRESENTLY\\21_22-j-02\\Presently\\presently\\media\\video\\22\\publicspeech.mp4")
    clip.audio.write_audiofile(r"F:\\CDAP-PRESENTLY\\21_22-j-02\\Presently\\presently\\users\\myprosody\\dataset\\audioFiles\\publicspeech.wav")
    r = sr.Recognizer()
    audio =sr.AudioFile(r"F:\\CDAP-PRESENTLY\\21_22-j-02\\Presently\\presently\\users\\myprosody\\dataset\\audioFiles\\publicspeech.wav")
    with audio as source:
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        audio_file = r.record(source)  
    stt = r.recognize_google(audio_file)
    return stt

def grammerchecker(speechText):
    happy_tt = HappyTextClassification(load_path = "F:\\CDAP-PRESENTLY\\21_22-j-02\\Presently\\presently\\users\\models\\grammermodels\\")
    beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=500)
    happy_tt = HappyTextToText("T5", "t5-base")
    example_1 = speechText
    result_1 = happy_tt.generate_text(example_1, args=beam_settings)
    return(result_1.text)

speechText = convertText()
result = grammerchecker(speechText)
