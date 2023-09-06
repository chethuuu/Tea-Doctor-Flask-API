# Import necessary libraries
import tensorflow as tf
import tensorflow_io as tfio
from itertools import groupby

# Load your audio classification model
audio_model = tf.keras.models.load_model(r"D:\Python\python_project\tea_api\model_weights\audio_classifiyer")

# Define a function to load and preprocess an MP3 file
def load_mp3_16k_mono(filename):
    # Load the audio file using TensorFlow I/O
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    
    # Calculate the mono channel by averaging the stereo channels
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2
    
    # Get the sample rate and convert it to int64
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    
    # Resample the audio to a 16kHz sample rate
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

# Define a function to preprocess an audio sample
def preprocess_mp3(sample, index):
    sample = sample[0]
    
    # Zero-pad the audio sample to a specific length (48000 in this case)
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    
    # Compute the spectrogram of the audio
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# Define a function to preprocess the entire audio file
def preprocess(file_path):
    # Load and preprocess the audio
    wav = load_mp3_16k_mono(file_path)
    
    # Create a time series dataset from the audio
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(
        wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1
    )
    
    # Preprocess the audio slices
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    # Make predictions using the audio model
    yhat = audio_model.predict(audio_slices)
    
    # Post-process the model's predictions (thresholding)
    yhat = [1 if prediction > 0.99 else 0 for prediction in yhat]
    
    # Group consecutive predictions with the same value
    yhat = [key for key, group in groupby(yhat)]
    
    # Calculate the sum of predicted insect calls
    insect_calls = tf.math.reduce_sum(yhat).numpy()
    print(insect_calls)
    return insect_calls














# import tensorflow as tf 
# import tensorflow_io as tfio
# from itertools import groupby

# audio_model = tf.keras.models.load_model(r"D:\Python\python_project\tea_api\model_weights\audio_classifiyer")


# def load_mp3_16k_mono(filename):

#     res = tfio.audio.AudioIOTensor(filename)
#     tensor = res.to_tensor()
#     tensor = tf.math.reduce_sum(tensor, axis=1) / 2
#     sample_rate = res.rate
#     sample_rate = tf.cast(sample_rate, dtype=tf.int64)
#     wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
#     return wav


        
# def preprocess_mp3(sample, index):
#     sample = sample[0]
#     zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
#     wav = tf.concat([zero_padding, sample],0)
#     spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
#     spectrogram = tf.abs(spectrogram)
#     spectrogram = tf.expand_dims(spectrogram, axis=2)
#     return spectrogram



# def preprocess(file_path): 
#     wav = load_mp3_16k_mono(file_path)
#     audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
#     audio_slices = audio_slices.map(preprocess_mp3)
#     audio_slices = audio_slices.batch(64)
#     yhat = audio_model.predict(audio_slices)
#     yhat = [1 if prediction > 0.99 else 0 for prediction in yhat]
#     yhat = [key for key, group in groupby(yhat)]
#     insect_calls = tf.math.reduce_sum(yhat).numpy()
#     print(insect_calls)
#     return insect_calls
        

    

        





