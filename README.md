# ECG Arrhythmia Detection Using LSTM and MIT-BIH Database #
This project implements an ECG signal processing and arrhythmia detection pipeline using the MIT-BIH Arrhythmia Database and a Long Short-Term Memory (LSTM) neural network.

The system performs:

* ECG signal acquisition from the MIT-BIH database
* Noise removal and signal preprocessing
* Frequency-domain and time-domain analysis
* Detection of ECG characteristic waves (P, R, and T waves)
* Heart rate and RR interval calculation
* Automatic dataset generation using sliding windows
* Binary arrhythmia classification using a deep LSTM model

---------------

# Dataset #

The project uses the MIT-BIH Arrhythmia Database, available through the WFDB package.

Dataset source:
* MIT-BIH Arrhythmia Database
The ECG recordings are sampled at **360 Hz** and contain expert annotations identifying normal and abnormal heartbeats.

---------------

# Signal Processing Pipeline #

 1. ECG Acquisition

The ECG recordings are loaded using the WFDB library:

```python
record = wfdb.rdrecord(...)
annotation = wfdb.rdann(..., "atr")
```

note: Only the MLII lead is used throughout the project.

---------------

 2. Noise Removal

Three filters are applied sequentially:


* High-Pass Filter:
Removes baseline wander
Butterworth filter
Order: 4
Cutoff frequency: 0.5 Hz



* Low-Pass Filter
Removes high-frequency noise:
* Butterworth filter
* Order: 4
* Cutoff frequency: 50 Hz



* Notch Filter
Suppresses power-line interference:
* Center frequency: 60 Hz
* Quality factor: 30

---------------

 3. Frequency Domain Analysis

The effectiveness of filtering is evaluated using:
* Fast Fourier Transform (FFT)
* Power Spectral Density (PSD)

Plots compare the signal before and after preprocessing.

---------------

 4. Time Domain Analysis

The project detects:
* R peaks using `scipy.signal.find_peaks`
* P-wave locations
* T-wave locations

Additionally, the following cardiac parameters are calculated:

* RR intervals
* Heart rate (BPM)

Visualization plots are generated for:

* Raw ECG signal
* Filtered ECG signal
* Detected ECG waves
* Heart rate variation
* RR interval variation

---------------

# Dataset Preparation #

Window Segmentation:
The filtered ECG is divided into overlapping windows:
* Window length: 1 second (360 samples)
* Overlap: 50%

Each window is assigned a label:

| Label | Description        |
| ----- | ------------------ |
| 0     | Normal rhythm      |
| 1     | Arrhythmia present |

A window is considered abnormal if at least one non-normal heartbeat annotation appears inside it.

---------------

 Normalization #

Each ECG signal undergoes:
1. Z-score normalization
2. StandardScaler normalization after train-test split
This improves model convergence and training stability.

---------------

# Deep Learning Model #

A stacked LSTM architecture is used for binary classification.

Architecture:

```text
Input (360 × 1)

LSTM (64)
|
Dropout (0.2)

LSTM (256)
|
Dropout (0.2)

LSTM (100)
|
Dropout (0.2)

Dense (1, Sigmoid)
```

# Training Configuration #

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Optimizer        | Adam                     |
| Loss Function    | Mean Squared Error (MSE) |
| Epochs           | 5                        |
| Batch Size       | 64                       |
| Validation Split | 20%                      |

---------------

# Evaluation Metrics #

The model performance is evaluated using:
* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-Score

Generated outputs include:

```python
confusion_matrix()
classification_report()
```

---------------

# Required Libraries #

```bash
pip install wfdb
pip install numpy pandas matplotlib scipy scikit-learn tensorflow
```
---------------

# Future Improvements 

Possible enhancements include:

* Multi-class heartbeat classification
* CNN-LSTM hybrid architecture
* Real-time ECG monitoring

---------------

Anoosha Sameti
Biomedical Engineering Student

ECG Signal Processing and Arrhythmia Detection using Deep Learning.
