from pyannote.audio import Model, Inference, Pipeline
import os, time, subprocess, torch, platform
from dotenv import load_dotenv
import speech_recognition as sr
from scipy.spatial.distance import cdist
import numpy as np 
from colorama import Fore, Style, init

init() # so that windows can detect the colors

load_dotenv() 

class WhisperSTT:
    def __init__(self, user_name, inference_file, user_num):
        self.device = torch.device("cpu")   # Change to "cuda" if you have an NVIDIA GPU
        embeddingmodel = Model.from_pretrained("pyannote/embedding", cache_dir="models/pyannote", 
                                               use_auth_token=os.getenv("HF_API_KEY"))
        
        self.inference = Inference(embeddingmodel, window="whole", device=self.device)
        self.main_speaker_embedding = self.inference(inference_file)
        self.user_name = user_name
        self.user_num = user_num
        self.color = Fore.YELLOW
        if self.user_num == 1:
            self.color = Fore.CYAN

    def callback(self, recognizer, audio):

        try:
            transcribed_file =  os.path.join("transcribed_audio", f"user{self.user_num}.wav")
            with open(transcribed_file, "wb") as f:
                f.write(audio.get_wav_data())

            start = time.time()
            output = self.process_audio(transcribed_file)
            end = time.time()
            clean_output = str(output).replace(" ", "").replace(".", "").replace(",", "").lower()
            if clean_output != "you" and clean_output != "":
                print("(Whisper)", self.color + f"{self.user_name} said: {output}" + Style.RESET_ALL)

        except sr.UnknownValueError:
            print("Speech Recognition could not understand you")

        except sr.RequestError as e:
            print('Could not request results from Google speech recognition service ')

    def listen(self):
        print(self.color + f"{self.user_name} Listening..." + Style.RESET_ALL)

        try:
            r = sr.Recognizer()
            with sr.Microphone(sample_rate=16000) as source:
                if self.user_num == 0:
                    time.sleep(0.5)
                print(self.color + f"{self.user_name} Say something" + Style.RESET_ALL)
            stop_listening = r.listen_in_background(source, self.callback)
            while True:
                time.sleep(0.05)

        except Exception as e:
            print(f"Error while listening: {e}")
            return False
        

    def speaker_verified(self, speaker_wav):
        speaker_embedding = self.inference(speaker_wav)
        distance = cdist(np.reshape(self.main_speaker_embedding, (1, -1)), np.reshape(speaker_embedding, (1, -1)), metric="cosine")[0, 0]

        print(self.color + f"{self.user_name} Speaker cosine distance {distance}" + Style.RESET_ALL)
        if distance < 0.675:
            return True
        
        return False
    
    def process_audio(self, wav_file, model_name="base.en"):

        if not self.speaker_verified(wav_file):
            # print("Speaker not verified")
            return ""
        
        model = os.path.join("modules", "whisper.cpp", "models", f"ggml-{model_name}.bin")

        if not os.path.exists(model):
            raise FileNotFoundError(f"Model not found: {model} \n\nDownload a model with this command \n\n> bash ./model/download-ggml-model-sh {model_name}\n\n")
        
        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found {wav_file}")
        
        system = platform.system()

        if system == "Windows":
            executable = os.path.abspath(
                os.path.join("modules", "whisper.cpp", "build", "bin", "Release", "whisper-cli.exe")
            )
        else:  # macOS or Linux
            executable = os.path.abspath(
                os.path.join("modules", "whisper.cpp", "main")
            )

        full_command = [executable, "-m", model, "-f", wav_file, "-np", "-nt", "-fa", "-l", "en"]
        process = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            raise Exception(f"Error parsing audio: {error.decode('utf-8')}")
        
        decoded_str = output.decode("utf-8").strip()
        processed_str = decoded_str.replace('[BLANK_AUDIO]', "").replace("[BLANK_AUDIO] ,", "").replace("[INAUDIBLE]", "").replace('[SILENCE]', "").replace('[SILENCE] ,', '').replace('[INAUDIBLE] ,', "").strip()

        return processed_str