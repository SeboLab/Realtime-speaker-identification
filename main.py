from dotenv import load_dotenv
from stt import WhisperSTT
import threading, os, time

load_dotenv()

voiceprint_files = os.listdir("voiceprint_audio")

if len(voiceprint_files) != 2:
    raise ValueError("There needs to be exactly two 15ish second videos of each user individually talking in voiceprint_audio")

user1_name = voiceprint_files[0].split(".")[0].capitalize()
user2_name = voiceprint_files[1].split(".")[0].capitalize()

print(os.path.join("voiceprint_audio", voiceprint_files[0]))

user1_STT = WhisperSTT(user1_name, os.path.join("voiceprint_audio", voiceprint_files[0]), 0)
user2_STT = WhisperSTT(user2_name, os.path.join("voiceprint_audio", voiceprint_files[1]), 1)
user1_thread = threading.Thread(target=user1_STT.listen)
user2_thread = threading.Thread(target=user2_STT.listen)

user1_thread.start()
time.sleep(1)
user2_thread.start()

user1_thread.join()
user2_thread.join()