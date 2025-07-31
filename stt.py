from pyannote.audio import Model, Inference
import os, time, torch, threading, pyaudio, queue, wave
from dotenv import load_dotenv
from scipy.spatial.distance import cdist
import numpy as np 
from colorama import Fore, Style, init
from check_mic_index import check_mic_idx
from microphone import begin_streaming

init() # so that Windows can detect the colors

load_dotenv() 

class WhisperSTT:
    def __init__(self, user_names, inference_files, user_num, barrier):
        self.device = torch.device("cpu")   # Change to "cuda" if you have an NVIDIA GPU
        embeddingmodel = Model.from_pretrained("pyannote/embedding", cache_dir="models/pyannote", 
                                               use_auth_token=os.getenv("HF_API_KEY"))
        
        self.inference = Inference(embeddingmodel, window="whole", device=self.device)
        self.user1_speaker_embedding = self.inference(inference_files[0])
        self.user2_speaker_embedding = self.inference(inference_files[1])

        self.user1 = user_names[0]
        self.user2 = user_names[1]
        self.user1_color = Fore.YELLOW
        self.user2_color = Fore.CYAN
        self.current_speaker = [None]
        self.audio_queue = queue.Queue()
        self.rate = 16000
        self.chunk = 1600
        check_mic_idx()
        self.device_index = int(input("Which device ID would you like to use?: "))
        self.revai_result = []

    def callback(self, recognizer, audio):
        transcribed_file =  os.path.join("transcribed_audio", "user.wav")
        with wave.open(transcribed_file, 'wb') as wf:
            wf.setnchannels(1)           # mono
            wf.setsampwidth(2)           # 16-bit PCM = 2 bytes
            wf.setframerate(16000)       # sample rate
            wf.writeframes(audio)

        self.current_speaker[0] = self.speaker_verified(transcribed_file)

    
    def mic_stream(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=self.rate,
                         input=True,
                         frames_per_buffer=self.chunk,
                         input_device_index=self.device_index)
        while True:
            data = stream.read(self.chunk, exception_on_overflow=False)
            self.audio_queue.put(data)

    def process_chunks(self):
        buffer = b""
        chunk_size =  self.rate * 4
        while True:
            data = self.audio_queue.get()
            buffer += data
            if len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                self.callback(None, chunk)


    def listen(self):

        try:
            threading.Thread(target=self.mic_stream, daemon=True).start()
            threading.Thread(target=self.process_chunks, daemon=True).start()
            time.sleep(0.5)
            revai_thread = threading.Thread(target=begin_streaming, args=(self.revai_result, self.device_index, self.current_speaker))
            revai_thread.start()

            print("waiting on rev")
            while True:
                if len(self.revai_result):
                    output = self.revai_result.pop()
                    color = self.user1_color
                    if self.current_speaker[0] == self.user2:
                        color = self.user2_color
                    
                    print("(RevAi)", color + f"{self.current_speaker[0]}" + Style.RESET_ALL, f"said: {output}" )
                    self.current_speaker[0] = None
                
                time.sleep(0.05)

        except Exception as e:
            print(f"Error while listening: {e}")
            return False
        

    def speaker_verified(self, speaker_wav):
        speaker_embedding = self.inference(speaker_wav)
        user1_distance = cdist(np.reshape(self.user1_speaker_embedding, (1, -1)), np.reshape(speaker_embedding, (1, -1)), metric="cosine")[0, 0]
        user2_distance = cdist(np.reshape(self.user2_speaker_embedding, (1, -1)), np.reshape(speaker_embedding, (1, -1)), metric="cosine")[0, 0]

        print(self.user1_color + f"{self.user1}" + Style.RESET_ALL, f"Speaker cosine distance {user1_distance}" )
        print(self.user2_color + f"{self.user2}" + Style.RESET_ALL, f"Speaker cosine distance {user2_distance}")

        if user1_distance < 0.75 and user2_distance < 0.75:
            if user1_distance < user2_distance:
                return self.user1
            return self.user2
        
        if user1_distance < 0.75:
            return self.user1
        
        if user2_distance < 0.75:
            return self.user2
        
        return None
    