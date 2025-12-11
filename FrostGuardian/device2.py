import sounddevice as sd
import soundfile as sf
import numpy as np
import subprocess


DURATION = 2.0
SR = 16000
OUT = 'demo_record.wav'


print('Recording...')
rec = sd.rec(int(DURATION*SR), samplerate=SR, channels=1)
sd.wait()
rec = rec.squeeze()
sf.write(OUT, rec, SR)
print('Saved', OUT)


# call inference
subprocess.run(['python3','inference.py',OUT])
