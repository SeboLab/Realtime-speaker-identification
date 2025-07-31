from pyannote.audio import Model, Inference, Pipeline
import os, time, subprocess, torch, platform, json, threading, pyaudio, queue, wave
from dotenv import load_dotenv
import speech_recognition as sr
from scipy.spatial.distance import cdist
import numpy as np 
from colorama import Fore, Style, init
from rev_ai import apiclient, JobStatus
from pydub import AudioSegment
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

        self.client = apiclient.RevAiAPIClient(os.getenv("REVAI_API_KEY"))
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

        try:
            transcribed_file =  os.path.join("transcribed_audio", "user.wav")
            # with open(transcribed_file, "wb") as f:
            #     f.write(audio.get_wav_data())
            with wave.open(transcribed_file, 'wb') as wf:
                wf.setnchannels(1)           # mono
                wf.setsampwidth(2)           # 16-bit PCM = 2 bytes
                wf.setframerate(16000)       # sample rate
                wf.writeframes(audio)

            self.current_speaker[0] = self.speaker_verified(transcribed_file)

            # output, current_speaker = self.process_audio(transcribed_file)
            # clean_output = str(output).replace(" ", "").replace(".", "").replace(",", "").lower()
            # color = self.user1_color
            # if current_speaker == self.user2:
            #     color = self.user2_color
            # if clean_output != "you" and clean_output != "":
            #     print("(RevAi)", color + f"{current_speaker}" + Style.RESET_ALL, f"said: {output}" )

        except sr.UnknownValueError:
            print("Speech Recognition could not understand you")

        except sr.RequestError as e:
            print('Could not request results from Google speech recognition service ')

    
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
            # r = sr.Recognizer()
            # r.energy_threshold = 100
            # with sr.Microphone(sample_rate=16000, device_index=self.device_index) as source:
            #     print("Say something")

            # stop_listening = r.listen_in_background(source, self.callback)
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
    
    def process_audio(self, wav_file, model_name="base.en"):
        
        current_speaker = self.speaker_verified(wav_file)

        if not current_speaker:
            return "", None
        
        audio = AudioSegment.from_file(wav_file, format="wav")
        duration = len(audio) / 1000.0
        difference = 2.1 - duration
        if difference > 0:
            silence = AudioSegment.silent(duration=difference*1000)
            padded_audio = audio + silence
        
        else:
            padded_audio = audio


        padded_audio_file = os.path.join("transcribed_audio", "padded_output.wav")
        padded_audio.export(padded_audio_file, format="wav")
        
        job = self.client.submit_job_local_file(padded_audio_file)
        job_details = self.client.get_job_details(job.id)

        start_time = time.time()
        while job_details.status == JobStatus.IN_PROGRESS:
            time.sleep(0.5)
            job_details = self.client.get_job_details(job.id)
        print(f"Revai finished in {time.time() - start_time} seconds")

        if job_details.status == JobStatus.FAILED:
            raise RuntimeError(f"Rev.ai job failed: {job_details.failure_detail}")

        transcript = self.client.get_transcript_json(job.id)

        text = "".join(e["value"] for mono in transcript["monologues"] for e in mono["elements"])

        return text, current_speaker
        
        # model = os.path.join("modules", "whisper.cpp", "models", f"ggml-{model_name}.bin")

        # if not os.path.exists(model):
        #     raise FileNotFoundError(f"Model not found: {model} \n\nDownload a model with this command \n\n> bash ./model/download-ggml-model-sh {model_name}\n\n")
        
        # if not os.path.exists(wav_file):
        #     raise FileNotFoundError(f"WAV file not found {wav_file}")
        
        # system = platform.system()

        # if system == "Windows":
        #     executable = os.path.abspath(
        #         os.path.join("modules", "whisper.cpp", "build", "bin", "Release", "whisper-cli.exe")
        #     )
        # else:  # macOS or Linux
        #     executable = os.path.abspath(
        #         os.path.join("modules", "whisper.cpp", "main")
        #     )

        # full_command = [executable, "-m", model, "-f", wav_file, "-np", "-nt", "-fa", "-l", "en"]
        # process = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # output, error = process.communicate()

        # if error:
        #     raise Exception(f"Error parsing audio: {error.decode('utf-8')}")
        
        # decoded_str = output.decode("utf-8").strip()
        # processed_str = decoded_str.replace('[BLANK_AUDIO]', "").replace("[BLANK_AUDIO] ,", "").replace("[INAUDIBLE]", "").replace('[SILENCE]', "").replace('[SILENCE] ,', '').replace('[INAUDIBLE] ,', "").strip()

        # return processed_str