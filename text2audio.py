import subprocess
import torch
import whisper
import datetime
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler,normalize
from scipy.io import wavfile
import noisereduce as nr
from pydub import AudioSegment
import locale
locale.getpreferredencoding = lambda: "UTF-8"

class TranscriptionProcessor:
    def __init__(self, num_speakers=2, language="any", model_size="large-v2"):
        self.num_speakers = num_speakers
        self.language = language
        self.model_size = model_size
        self.audio = Audio()
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            #"speechbrain/asr-crdnn-commonvoice-14-es",
            device=torch.device("cuda")
        )
        self.model = whisper.load_model(self.model_size)

    def convert_to_mono(self):
        audio_estereo = AudioSegment.from_file(self.audio_path)
        audio_mono = audio_estereo.set_channels(1)
        audio_mono.export("/workspace/mono.wav", format="wav")

    def reduce_noise(self):
        rate, data = wavfile.read("/workspace/mono.wav")
        reduced_noise = nr.reduce_noise(y=data, sr=rate)
        wavfile.write("/workspace/mono_reduced.wav", rate, reduced_noise)

    def load_whisper_model(self):
        #subprocess.call(['ffmpeg', '-i', '/content/mono_reduced.wav', '/content/mono.wav', '-y'])
        result = self.model.transcribe("/workspace/mono_reduced.wav")
        self.segments = result["segments"]

    def transcribe_audio(self):
        frames, rate, duration = self.get_audio_info('/workspace/mono_reduced.wav')
        self.embeddings = np.zeros(shape=(len(self.segments), 192))
        for i, segment in enumerate(self.segments):
            self.embeddings[i] = self.segment_embedding(segment, duration)

        self.embeddings = np.nan_to_num(self.embeddings)

    def segment_embedding(self, segment, duration):
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, _ = self.audio.crop('/workspace/mono_reduced.wav', clip)
        return self.embedding_model(waveform[None])

    def cluster_speakers(self):
        clustering = AgglomerativeClustering(
          self.num_speakers,
          metric='euclidean',
          linkage='ward').fit(normalize(self.embeddings))
        #clustering = AgglomerativeClustering(
        #    self.num_speakers, metric='euclidean', linkage='ward'
        #).fit(MinMaxScaler().fit_transform(self.embeddings))
        self.labels = clustering.labels_
        for i in range(len(self.segments)):
            self.segments[i]["speaker"] = 'SPEAKER ' + str(self.labels[i] + 1)

    def save_transcription(self):
        with open("/workspace/transcript.txt", "w", encoding="utf-8") as f:
            for (i, segment) in enumerate(self.segments):
                if i == 0 or self.segments[i - 1]["speaker"] != segment["speaker"]:
                    f.write("\n" + segment["speaker"] + ',' + str(self.time(segment["start"])) + ',')
                f.write(segment["text"][1:] + ' ')

    def time(self, secs):
        return datetime.timedelta(seconds=round(secs))

    def get_audio_info(self, filepath):
        with contextlib.closing(wave.open(filepath, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        return frames, rate, duration

    def run_transcription(self,audio_path):
        self.audio_path = audio_path
        self.convert_to_mono()
        self.reduce_noise()
        self.load_whisper_model()
        self.transcribe_audio()
        self.cluster_speakers()
        self.save_transcription()