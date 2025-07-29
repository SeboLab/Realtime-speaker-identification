import torch
from pyannote.audio import Model, Inference, Pipeline
import os, time
from dotenv import load_dotenv
import speech_recognition as sr
from scipy.spatial.distance import cdist
import numpy as np
import subprocess   


load_dotenv()

class WhisperSTT:
    def __init__(self):
        self.device = torch.device("cpu")
        embeddingmodel = Model.from_pretrained("pyannote/embedding", cache_dir="models/pyannote", 
                                               use_auth_token=os.getenv("HF_API_KEY"))
        self.inference = Inference(embeddingmodel, window="whole", device=self.device)

        self.main_speaker_embedding = self.inference('brian.wav')

    def callback(self, recognizer, audio):

        try:
            transcribed_file =  "transcribed_audio.wav"
            with open(transcribed_file, "wb") as f:
                f.write(audio.get_wav_data())

            start = time.time()
            output = self.process_audio(transcribed_file)
            end = time.time()
            print(f"Listen offline transcription in {end - start} seconds")
            clean_output = str(output).replace(" ", "").replace(".", "").replace(",", "").lower()
            if clean_output != "you" and clean_output != "":
                print(f"(Whisper) you said: {output}")

        except sr.UnknownValueError:
            print("Speech Recognition could not understand you")

        except sr.RequestError as e:
            print('Could not request results from Google speech recognition service ')

    def listen(self):
        print("Listening...")

        try:
            r = sr.Recognizer()
            with sr.Microphone(sample_rate=16000) as source:
                print("Say something")
            stop_listening = r.listen_in_background(source, self.callback)
            while True:
                time.sleep(0.05)

        except Exception as e:
            print(f"Error while listening: {e}")
            return False
        

    def speaker_verified(self, speaker_wav):
        speaker_embedding = self.inference(speaker_wav)
        distance = cdist(np.reshape(self.main_speaker_embedding, (1, -1)), np.reshape(speaker_embedding, (1, -1)), metric="cosine")[0, 0]

        print("Speaker cosine distance", distance)
        if distance < 0.65:
            return True
        
        return False
    
    def process_audio(self, wav_file, model_name="base.en"):

        if not self.speaker_verified(wav_file):
            print("Speaker not verified")
            return ""
        
        model = os.path.join("modules", "whisper.cpp", "models", f"ggml-{model_name}.bin")

        if not os.path.exists(model):
            raise FileNotFoundError(f"Model not found: {model} \n\nDownload a model with this command \n\n> bash ./model/download-ggml-model-sh {model_name}\n\n")
        
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found {wav_file}")
        
        executable = os.path.abspath(
            os.path.join("modules", "whisper.cpp", "build", "bin", "Release", "whisper-cli.exe")
        )
        full_command = [executable, "-m", model, "-f", wav_file, "-np", "-nt", "-fa", "-l", "en"]

        process = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)




        output, error = process.communicate()

        if error:
            raise Exception(f"Error parsing audio: {error.decode('utf-8')}")
        
        decoded_str = output.decode("utf-8").strip()

        processed_str = decoded_str.replace('[BLANK_AUDIO]', "").replace("[BLANK_AUDIO] ,", "").replace("[INAUDIBLE]", "").replace('[SILENCE]', "").replace('[SILENCE] ,', '').replace('[INAUDIBLE] ,', "").strip()

        return processed_str