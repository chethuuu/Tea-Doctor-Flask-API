import tensorflow as tf 
import tensorflow_io as tfio
from itertools import groupby

audio_model = tf.keras.models.load_model(r"D:\Python\python_project\tea_api\model_weights\audio_classifiyer")


def load_mp3_16k_mono(filename):

    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav


        
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram



def preprocess(file_path): 
    wav = load_mp3_16k_mono(file_path)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    yhat = audio_model.predict(audio_slices)
    yhat = [1 if prediction > 0.99 else 0 for prediction in yhat]
    yhat = [key for key, group in groupby(yhat)]
    insect_calls = tf.math.reduce_sum(yhat).numpy()
    print(insect_calls)
    return insect_calls
        

    

        





