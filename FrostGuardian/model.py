import librosa
import numpy as np


def wav_to_mel(path, sr=16000, n_mels=64, duration=2.0):
y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
mel_db = librosa.power_to_db(mel)
# normalize
mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
return mel_db.astype(np.float32)


if __name__ == '__main__':
import sys
arr = wav_to_mel(sys.argv[1])
np.save('sample_mel.npy', arr)
print('Saved sample_mel.npy', arr.shape)

import tensorflow as tf
import numpy as np


# tiny Keras model matching the PyTorch architecture
model = tf.keras.Sequential([
tf.keras.layers.Input(shape=(64, 32, 1)),
tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
tf.keras.layers.MaxPool2D(2),
tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
tf.keras.layers.GlobalAveragePooling2D(),
tf.keras.layers.Dense(3, activation='softmax')
])


# pretend we trained; here we just save and convert the untrained model as a demo
model.save('tf_model')
converter = tf.lite.TFLiteConverter.from_saved_model('tf_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


# representative dataset generator
def representative_data_gen():
for _ in range(100):
data = np.random.rand(1,64,32,1).astype(np.float32)
yield [data]


converter.representative_dataset = representative_data_gen


tflite_model = converter.convert()
open('model_quant.tflite','wb').write(tflite_model)
print('Saved model_quant.tflite')


