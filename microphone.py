import pyaudio
from rev_ai.models import MediaConfig
from rev_ai.streamingclient import RevAiStreamingClient
import os, subprocess, json, time, queue, string
from dotenv import load_dotenv
import numpy as np

load_dotenv()

access_token = os.getenv("REVAI_API_KEY")

class MicrophoneStream(object):
    def __init__(self, rate, chunk, device_idx):
        self._rate = rate
        self._chunk = chunk
        self._device_idx = device_idx
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            input_device_index=self._device_idx,
            stream_callback=self._fill_buffer,
        )
        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue
    
    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            
            raw_data = b''.join(data)

            yield raw_data


rate = 44100
chunk = int(rate / 10)

example_mc = MediaConfig("audio/x-raw", "interleaved", 44100, "S16LE", 1)
streamclient = RevAiStreamingClient(access_token, example_mc)

def append_text(text, last_partial, result, speaker, current_phrase, current_speaker):
    text = text.split(last_partial)[-1]
    text = text.strip()
    if text != "" and current_speaker[0]:
        last_partial = text
        if current_speaker[0] != speaker:
            result[0] = f"{speaker} said:" + "".join(current_phrase)
            current_phrase.clear()
        current_phrase.append(text)
    return last_partial

def begin_streaming(result, device_idx, current_speaker):
    with MicrophoneStream(rate, chunk, device_idx) as stream:
        try:    
            response_gen = streamclient.start(stream.generator())
            # revai_speaker = None
            # last_partial = " "
            # full_phrase = []
            for response in response_gen:
                response = json.loads(response)
                if response['type'] == 'final':
                    text = "".join(e['value'] for e in response['elements'])
                    if current_speaker[0]:
                        result.append(text)
                    

        except KeyboardInterrupt:
            streamclient.end()
            pass

        except Exception as e:
            print(f"Error while streaming: {e}")
            streamclient.end()
