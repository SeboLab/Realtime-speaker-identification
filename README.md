# Realtime-speaker-identification

## What is it?

Uses pyannote and Whisper to get realtime transcriptions along with speaker lables through a streaming microphone

## Setup - For Everyone

### 1. pip install all requirements from requirements.txt
I would recommend running on Python 3.12.0, other versions may produce conflicts

### 2. Create a folder called models
- Within it create a subfolder called pyannote
- This will be used to store the embedding that pyannote creates

### 3. Create a HuggingFace account and get an access token
- Create your account and go [here](https://huggingface.co/pyannote/embedding) to get access to pyannote models
- Then click on your profile on the top-right and make yourself an access token with write permissions
- Add this token to a `.env` file, and label it `HF_API_KEY=[your HF access token]`

### 4. Create another folder called modules
- Within this folder, run `git clone https://github.com/ggerganov/whisper.cpp.git`
- For linux/Mac OS this is all 

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



