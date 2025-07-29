# realtime-speaker-identification

## What is it?

Uses pyannote and Whisper to get realtime transcriptions along with speaker lables through a streaming microphone

## Setup - For Everyone

I would recommend running on Python 3.12.0, other versions may produce conflicts

### pip install all requirements from requirements.txt

### Create a folder called models
- Within it create a subfolder called pyannote
- This will be used to store the embedding that pyannote creates

### Create a HuggingFace account and get an access token
- Create your account and go to [here](https://huggingface.co/pyannote/embedding) to get access to pyannote models
- Then click on your profile on the top-right and make yourself an access token with write permissions
- Add this token to a `.env` file, and label it `HF_API_KEY=[your HF access token]`

### Create another folder called modules
- Within this folder, run `git clone https://github.com/ggerganov/whisper.cpp.git`
- For linux/Mac OS this is all 

## Setup - For Windows

### Install Cmake at this [link](https://cmake.org/download/)
- Download the Windows 64 installer and run

### Run the following commands in the modules folder
```
cd whisper.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```


