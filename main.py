from dotenv import load_dotenv
from stt import WhisperSTT

load_dotenv()

wSTT = WhisperSTT()
wSTT.listen()