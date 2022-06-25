import torch
from scipy.fft import fft, ifft
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vlc
import librosa
import librosa.display

# %%
print("Hello World")
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
tensor = torch.rand(3,4)
print(torch.cuda.device_count())
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %%
audio_array, sr = librosa.load(r"C:\Users\cwebs\Downloads\MediaHuman\Music\Cat_Mouse.mp3")
print(audio_array.shape)
print(sr)
#librosa.display.waveshow(audio_array,sr)
# %%
freqdom = fft(audio_array)
# %%
plt.plot(freqdom)
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()

# %%
def plot_mp3_matplot(filename):
    """
    plot_mp3_matplot -- using matplotlib to simply plot time vs amplitude waveplot
    
    Arguments:
    filename -- filepath to the file that you want to see the waveplot for
    
    Returns -- None
    """
    
    # sr is for 'sampling rate'
    # Feel free to adjust it
    x, sr = librosa.load(filename, sr=44100)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)
# %%
def convert_audio_to_spectrogram(filename):
    """
    convert_audio_to_spectrogram -- using librosa to simply plot a spectogram
    
    Arguments:
    filename -- filepath to the file that you want to see the waveplot for
    
    Returns -- None
    """
    
    # sr == sampling rate 
    x, sr = librosa.load(filename, sr=44100)
    
    # stft is short time fourier transform
    X = librosa.stft(x)
    
    # convert the slices to amplitude
    Xdb = librosa.amplitude_to_db(abs(X))
    
    # ... and plot, magic!
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'hz')
    plt.colorbar()

# %%
def convert_audio_to_spectrogram_log(filename):
    x, sr = librosa.load(filename, sr=44100)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis = 'log')
    plt.colorbar()

# %%
plot_mp3_matplot(r"C:\Users\cwebs\Downloads\MediaHuman\Music\Cat_Mouse.mp3")
# %%
convert_audio_to_spectrogram(r"C:\Users\cwebs\Downloads\MediaHuman\Music\Cat_Mouse.mp3")
# %%
convert_audio_to_spectrogram_log(r"C:\Users\cwebs\Downloads\MediaHuman\Music\Cat_Mouse.mp3")
# %%
