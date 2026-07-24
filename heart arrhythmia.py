# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1BflESNwFrlzKHi2wbOFxLmQdzHLwFo6P
    
# **`Libraries`**
"""

#!pip install wfdb

import wfdb 
download=wfdb.dl_database('mitdb','mitdb')

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.ndimage as ndimage
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

"""# **`Checking out the data`**"""

def choose_record():
    no = {110, 120, 204, 206, 211, 216, 218, 229}
    records = []
    for i in range(100, 235):
        if i not in no:
            records.append(f"mitdb/{i}")
    answer = input(
        "Random record or specific number?\n"
        "Type 'Random' or choose from 100–234 (dont choose 110 120 204 206 211 216 218 229): ")

    if answer.lower() == 'random':
        x = random.choice(records)
        print(f"You are using {x}")
        return x
    try:
        num = int(answer)
    except ValueError:
        print("Invalid input. Using random record instead.")
        return random.choice(records)

    if num < 100 or num > 234 or num in no:
        print("Invalid record number. Using random record instead.")
        return random.choice(records)

    x = f"mitdb/{num}"
    print(f"You are using {x}")
    return x

x = choose_record()
record = wfdb.rdrecord(x, channel_names=['MLII'])
annotation = wfdb.rdann(x, "atr")

signal=record.p_signal.flatten()
Fs= int(record.fs)
N = len(signal)
f = np.arange(N) * (Fs / N)
Y = np.fft.fft(signal)

ann_samples = np.array(annotation.sample)
ann_symbols = np.array(annotation.symbol)

# Baseline wander removal / high pass
b, a = sig.butter(4, 0.5, 'high', fs=Fs, analog=False)
#filtering the signal and removing ndarray a and b from it
ecg_preprocessed = sig.filtfilt(b, a, signal)

#low pass filter
b, a= sig.butter(4 , 50 , 'low' , fs=Fs, analog=False)
ecg_preprocessed = sig.filtfilt(b, a, ecg_preprocessed)

# Power line interference removal
b, a = sig.iirnotch(60, 30, fs=Fs)
ecg_preprocessed = sig.filtfilt(b, a, ecg_preprocessed)
Y_filtered = np.fft.fft(ecg_preprocessed)

"""# **`Plot in freq domain`**"""

#FREQ DOMAIN

#Spectrum Before Noise Removal
plt.figure(figsize=(12, 8))
plt.subplot (2,1,1)
plt.plot(f[:N//2], np.abs(Y[:N//2]))
plt.title('Spectrum Before Power Noise Removal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.ylim(-10 , 7000)
plt.xlim (0,100)

#Spectrum After Noise Removal
plt.subplot (2,1,2)
plt.plot(f[:N//2], np.abs(Y_filtered[:N//2]))
plt.title('Spectrum After Power Noise Removal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.ylim(-10 ,7000)
plt.xlim (0,100)
plt.tight_layout()
plt.show()

#Power Spectral Density Before
(ff, psd) = sig.periodogram(signal, fs=Fs, scaling='density')
plt.semilogy(ff, psd)
plt.ylim([1e-8, 1e2])
plt.xlim([0,100])
plt.title('Power Spectral Density Before')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

#Power Spectral Density After
(ff, psd) = sig.periodogram(ecg_preprocessed, fs=Fs, scaling='density')
plt.semilogy(ff, psd)
plt.ylim([1e-8, 1e2])
plt.xlim([0,100])
plt.title('Power Spectral Density After')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

"""# **`Plot in time domain`**"""

#TIME DOMAIN
peaks_r , info =sig.find_peaks(ecg_preprocessed , distance=150 , height = 0.65)
p_wave_peaks = []
t_wave_peaks = []
p_window_len = int(0.1 * Fs)
t_window_len = int(0.6 * Fs)

peak_time = peaks_r / Fs
r_r = np.diff(peaks_r) / Fs
heart_rate = 60 / r_r

for i in range(len(peaks_r)):
    current_rr_duration_samples = int(r_r[i-1] * Fs)

    p_start = peaks_r[i] - int(0.18 * current_rr_duration_samples)
    p_end   = peaks_r[i] - int(0.1 * current_rr_duration_samples)
    if p_start >= 0 and p_end < len(ecg_preprocessed) and p_start < p_end:
        p_wave_segment = ecg_preprocessed[p_start:p_end]
        if len(p_wave_segment) > 0:
            p_wave_id = np.argmax(p_wave_segment) + p_start
            p_wave_peaks.append(p_wave_id)

    t_start =  peaks_r[i] - int(0.9 * current_rr_duration_samples)
    t_end = peaks_r[i] - int(0.3 * current_rr_duration_samples)
    if t_start >= 0 and t_end < len(ecg_preprocessed) and t_start < t_end:
        t_wave_segment = ecg_preprocessed[t_start:t_end]
        if len(t_wave_segment) > 0:
            t_wave_id = np.argmax(t_wave_segment) + t_start
            t_wave_peaks.append(t_wave_id)

#original signal in time domain
plt.figure(figsize=(15, 5))
plt.plot(signal)
plt.xlim([0,1000])
plt.xlabel('sample time from 0 to 1000s')
plt.ylabel('amp')
plt.title('Original signal')
plt.grid(True)
plt.show()

#filtered signal with the peaks and wave
plt.figure(figsize=(15,5))
plt.plot(ecg_preprocessed, label= 'Filtered signal')
plt.plot(peaks_r, ecg_preprocessed[peaks_r], 'ro' , label='Detected R peaks')
plt.title('QRS COMPLEX')
plt.plot(p_wave_peaks, ecg_preprocessed[p_wave_peaks], 'bo', label='Detected P wave')
plt.plot(t_wave_peaks, ecg_preprocessed[t_wave_peaks], 'go', label='Detected T wave')
plt.legend()
plt.xlim([0,1000])
plt.xlabel('sample time from 0 to 1000s')
plt.grid(True)
plt.ylabel('amp')
plt.show()

plt.figure(figsize=(10,5))
plt.subplot(4,1,1)
plt.plot(peak_time[1:], heart_rate, 'r-', linewidth=2)
plt.xlim([0,50])
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate (BPM)')
plt.title('Heart Rate (BPM) inspected closely')
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(peak_time[1:], r_r, 'g-', label= 'R_R', linewidth=2)
plt.xlim([0,50])
plt.xlabel('Time (s)')
plt.ylabel('RR Interval (s)')
plt.title('RR Intervals inspected closely')
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(peak_time[1:], heart_rate, 'r-', linewidth=2)
plt.xlim([0,600])
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate (BPM)')
plt.title('Heart Rate (BPM)')
plt.grid(True)

plt.subplot(4,1,4)
plt.plot(peak_time[1:], r_r, 'g-', label= 'R_R', linewidth=2)
plt.xlim([0,600])
plt.xlabel('Time (s)')
plt.ylabel('RR Interval (s)')
plt.title('RR Intervals')
plt.grid(True)

plt.tight_layout()
plt.show()

"""# **`creating DB and Def`**"""

records = [
    "101","100","102","103","104","105","106","107","108","109","111",
    "112","113","114","115","116","117","118","119","121","122","123",
    "124","200","201","202","203","205","207","208","209","210","212",
    "213","215","217","219","220","221","222","228","233","232","231",
    "223","234"
]
def filter_ecg(ecg, fs):
    # High-pass
    b, a = sig.butter(4, 0.5, 'high', fs=fs)
    ecg = sig.filtfilt(b, a, ecg)

    # Low-pass
    b, a = sig.butter(4, 50, 'low', fs=fs)
    ecg = sig.filtfilt(b, a, ecg)

    # Notch 60Hz
    b, a = sig.iirnotch(60, 30, fs=fs)
    ecg = sig.filtfilt(b, a, ecg)

    return ecg

def zscore(ecg):
    ecg = np.asarray(ecg)
    mean = ecg.mean()
    std = ecg.std()
    return (ecg - mean) / (std + 1e-8)

def process(record_name):
    record = wfdb.rdrecord("mitdb/" + record_name, channel_names=['MLII'])
    if record is None or record.p_signal is None:
        return np.array([]).reshape(0, Fs, 1), np.array([])

    ann = wfdb.rdann("mitdb/" + record_name, "atr")
    if ann is None or not ann.sample.size:
        return np.array([]).reshape(0, Fs, 1), np.array([])

    ann_samples = np.array(ann.sample)
    ann_symbols = np.array(ann.symbol)

    ecg = record.p_signal.flatten()
    ecg_f = filter_ecg(ecg, Fs)
    ecg_n = zscore(ecg_f)

    beat_labels = np.array([0 if s == 'N' else 1 for s in ann_symbols])
    sample_labels = np.zeros(len(ecg_n), dtype=int)
    valid = ann_samples < len(sample_labels)
    sample_labels[ann_samples[valid]] = beat_labels[valid]

    window_size = Fs
    step = window_size // 2

    X_list = []
    y_list = []

    for i in range(0, len(ecg_n) - window_size, step):
        window = ecg_n[i:i+window_size].reshape(-1, 1)
        label = int(np.any(sample_labels[i:i+window_size] == 1))
        X_list.append(window)
        y_list.append(label)

    return np.array(X_list), np.array(y_list)

Xall = []
yall = []

for i in records:
    print("Processing record:", i)
    X, y = process(i)
    Xall.append(X)
    yall.append(y)

X = np.vstack(Xall)
y = np.hstack(yall)

print("Total windows:", X.shape)
print("Arrhythmia percentage:", np.mean(y))

"""# **`Start training`**"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

scaler = StandardScaler()

X_train_flat = X_train.reshape(-1, 1)
X_test_flat  = X_test.reshape(-1, 1)

scaler.fit(X_train_flat)

X_train = scaler.transform(X_train_flat).reshape(X_train.shape)
X_test  = scaler.transform(X_test_flat).reshape(X_test.shape)

def build_lstm_model(window_size=720):
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(100))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    return model


model = build_lstm_model()
model.summary()


history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=2)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy:", acc)
print("The loss:", loss)

y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.4).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
