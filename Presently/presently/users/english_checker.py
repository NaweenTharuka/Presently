import numpy as np
from colorama import Fore
from IPython.display import Audio
import librosa, enchant, os, torch
from IPython.display import display
from happytransformer import HappyTextToText, TTSettings
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModelForSeq2SeqLM

class EnglishChecker(object):
    def __init__(self):
        self.sampling_rate=16000
        self.en_US_vocab = enchant.Dict("en_US")
        self.en_UK_vocab = enchant.Dict("en_UK")
        
        self.ASR_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.ASR_MODEL = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

        self.HAPPY_TT = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
        self.BEAM_SETTINGS =  TTSettings(num_beams=5, min_length=1, max_length=200)
 
    def process_audio(self, audio_path):
        audio_wave = librosa.load(audio_path, sr=self.sampling_rate)[0]
        input_wave = self.ASR_PROCESSOR(
                                    audio_wave, 
                                    padding="longest",
                                    return_tensors="pt", 
                                    sampling_rate=self.sampling_rate
                                    ).input_values
        return input_wave

    def speech2text(self, audio_path):
        input_wave = self.process_audio(audio_path)
        logits = self.ASR_MODEL(input_wave).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.ASR_PROCESSOR.batch_decode(predicted_ids)  
        return transcription[0]

    def detect_mispronunciation(self, transcription):
        tokens = transcription.split()
        mispronunciation = [token for token in tokens if (not self.en_US_vocab.check(token)) and (not self.en_UK_vocab.check(token))]
        detected_transcription = ''
        misprounced_words = []
        for token in tokens:
            if token in mispronunciation:
                detected_transcription += Fore.RED + token + Fore.RESET + ' '
                misprounced_words.append(token)
            else:
                detected_transcription += token + ' '
        return detected_transcription

    def ASR(self, audio_path):
        transcription = self.speech2text(audio_path)
        detected_transcription = self.detect_mispronunciation(transcription)
        return transcription, detected_transcription

    def EGC(self, detected_transcription):
        result = self.HAPPY_TT.generate_text(
                                            detected_transcription, 
                                            args=self.BEAM_SETTINGS
                                            )
        return result.text

    def checker(self, audio_path):
        transcription, detected_transcription = self.ASR(audio_path)
        grammer_corrected = self.EGC(detected_transcription)
        return transcription, grammer_corrected