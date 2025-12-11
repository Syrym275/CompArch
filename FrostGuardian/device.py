import numpy as np
import soundfile as sf
import librosa
import tflite_runtime.interpreter as tflite


MODEL_PATH = 'model_quant.tflite'


def wav_to_mel_array(path, sr=16000, n_mels=64, duration=2.0):
y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
mel_db = librosa.power_to_db(mel)
mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
# ensure shape HxW (pad or crop)
mel_db = mel_db[:64,:32]
if mel_db.shape != (64,32):
pad = np.zeros((64,32), dtype=np.float32)
h,w = mel_db.shape
pad[:h,:w] = mel_db
mel_db = pad
return mel_db.astype(np.float32)




def load_model(path):
interpreter = tflite.Interpreter(model_path=path)
interpreter.allocate_tensors()
return interpreter




def infer(interpreter, input_tensor):
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
# if quantized, convert input
details = interpreter.get_input_details()[0]
if details['dtype'] == np.int8:
scale, zero_point = details['quantization']
input_tensor = (input_tensor/scale + zero_point).astype(np.int8)
interpreter.set_tensor(input_index, input_tensor)
interpreter.invoke()
out = interpreter.get_tensor(output_index)
return out


if __name__ == '__main__':
import sys
wav = sys.argv[1] if len(sys.argv)>1 else 'test.wav'
mel = wav_to_mel_array(wav)
x = mel.reshape(1,64,32,1)
interp = load_model(MODEL_PATH)
res = infer(interp, x)
print('Raw model output:', res)
