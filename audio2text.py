from pydub import AudioSegment
import sys
import whisper
import datetime
import subprocess
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import locale

def segment_embedding(segment):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  return embedding_model(waveform[None])

def time(secs):
  return datetime.timedelta(seconds=round(secs))
# Cargar el archivo de audio estéreo
path = sys.argv[1]
speakers = sys.argv[2]
lang = sys.argv[3]

# Convertir a mono
audio_estereo = AudioSegment.from_file(path)
audio_mono = audio_estereo.set_channels(1)

# Guardar el audio en formato mono
audio_mono.export(path, format="mp3")

# Set Number of speakers, Language & model
num_speakers = speakers #@param {type:"integer"}
language = lang #@param ["any", "English", "Dutch"]
model_size = "large-v2" #@param ["tiny", "base", "small", "medium", "large", "large-v2"]

model_name = model_size
if language == 'English' and model_size != 'large':
  model_name += '.en'

if path[-3:] != 'wav':
  subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
  path = 'audio.wav'
  model = whisper.load_model(model_size)

locale.getpreferredencoding = lambda: "UTF-8"
model = whisper.load_model(model_size)

result = model.transcribe(path)
segments = result["segments"]
subprocess.call(['ffmpeg', '-i', path, 'mono.wav','-y'])

with contextlib.closing(wave.open('mono.wav','r')) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames / float(rate)

audio = Audio()

embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
  embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_
for i in range(len(segments)):
  segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

f = open("output_files/transcript.txt", "w", encoding="utf-8")

for (i, segment) in enumerate(segments):
  if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
    f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
  f.write(segment["text"][1:] + ' ')
f.close()