from dotenv import load_dotenv
from stt import WhisperSTT
import os

load_dotenv()

nested_path = os.path.join("models", "pyannote")
os.makedirs(nested_path, exist_ok=True)
nested_path = os.path.join("transcribe_audio")
os.makedirs(nested_path, exist_ok=True)

voiceprint_files = os.listdir("voiceprint_audio")

if len(voiceprint_files) != 2:
    raise ValueError("There needs to be exactly two 15ish second videos of each user individually talking in voiceprint_audio")

user1_name = voiceprint_files[0].split(".")[0].capitalize()
user2_name = voiceprint_files[1].split(".")[0].capitalize()

print(os.path.join("voiceprint_audio", voiceprint_files[0]))

whisp = WhisperSTT([user1_name, user2_name], [os.path.join("voiceprint_audio", voiceprint_files[0]), os.path.join("voiceprint_audio", voiceprint_files[1])], None, None)
whisp.listen()