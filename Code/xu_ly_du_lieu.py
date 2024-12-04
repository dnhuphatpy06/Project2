from sklearn.ensemble import ExtraTreesClassifier
import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
import _pickle as pickle

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def extract_labels(file_name):
    label = file_name.split("_")[-2]
    return label

def parse_audio_files(path):
    features_list = []  
    labels_list = []   
    for fn in glob.glob(path + "\.wav"):
        mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features_list.append(ext_features)
        label = extract_labels(fn)
        labels_list.append(label)
    features = np.array(features_list)
    labels = np.array(labels_list)
    return features, labels
