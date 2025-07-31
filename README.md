# Realtime-speaker-identification

## What is it?

Uses pyannote and RevAi to get realtime transcriptions along with speaker lables for two users speaking through a single streaming microphone. Credit goes to this [youtube video](https://www.youtube.com/watch?v=uf5oth4-eF8) by Tech Giant. Code is largely based on his, but tweaked to work with two people at once and tweaked to work with RevAI

## Setup - For Everyone

### 1. pip install all requirements from requirements.txt
I would recommend running on Python 3.12.0, other versions may produce conflicts

### 2. Create a folder called voiceprint_audio
- Will hold the 15s audio clips of each speaker for pyannote to use to create your voiceprint

### 3. Create a HuggingFace account and get an access token
- Create your account and go [here](https://huggingface.co/pyannote/embedding) to get access to pyannote models
- Then click on your profile on the top-right and make yourself an access token with write permissions
- Add this token to a `.env` file, and label it `HF_API_KEY=[your HF access token]`

### 4. Add your RevAi API key into the `.env` file as well
```
REVAI_API_KEY=[your RevAI API key]
```

## How to run

### 1. Add your two voiceprint audio files to the voiceprint audio folder
- Each audio file should be around 15s, and should feature only your voice in a clear non-noisy envrionment
- Label them as `[name].wav` so that your speaker labels have names
- This works for two speakers ONLY, so there must be exactly two audio files here

### 2. Run main.py and that's it!
- Transcription along with speaker labels will be printed out into the terminal
- Print statements with cosine distance will also be printed, which just tells you how likely it was that you were speaking (>0.75 means unlikely, <0.75 means likley)
- For best performance, try not to speak at the same time


