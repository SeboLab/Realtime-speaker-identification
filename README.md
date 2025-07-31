# Realtime-speaker-identification

## What is it?

Uses pyannote and Whisper to get realtime transcriptions along with speaker lables for two users speaking through a single streaming microphone. Credit goes to this [youtube video](https://www.youtube.com/watch?v=uf5oth4-eF8) by Tech Giant. Code is largely based on his, but tweaked to work with two people at once

## Setup - For Everyone

### 1. pip install all requirements from requirements.txt
I would recommend running on Python 3.12.0, other versions may produce conflicts

### 2. Create a folder called voiceprint_audio
- Will hold the 15s audio clips of each speaker for pyannote to use to create your voiceprint

### 3. Create a HuggingFace account and get an access token
- Create your account and go [here](https://huggingface.co/pyannote/embedding) to get access to pyannote models
- Then click on your profile on the top-right and make yourself an access token with write permissions
- Add this token to a `.env` file, and label it `HF_API_KEY=[your HF access token]`

### 4. Create another folder called modules
- Within this folder, run `git clone https://github.com/ggerganov/whisper.cpp.git`
- Go into the whisper.cpp folder and run
`bash ./models/download-ggml-model.sh base.en`
- This will download the base english model from Whisper

## Setup - For Windows

### 5. Install Cmake at this [link](https://cmake.org/download/)
- Download the Windows 64 installer and run

### 6. Run the following commands in the modules folder
```
cd whisper.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## How to run

### 1. Add your two voiceprint audio files to the voiceprint audio folder
- Each audio file should be around 15s, and should feature only your voice in a clear non-noisy envrionment
- Label them as `[name].wav` so that your speaker labels have names
- This works for two speakers ONLY, so there must be exactly two audio files here

### 2. Run main.py and that's it!
- Transcription along with speaker labels will be printed out into the terminal
- Print statements with cosine distance will also be printed, which just tells you how likely it was that you were speaking (>0.675 means unlikely, <0.675 means likley)
- For best performance, try not to speak at the same time


