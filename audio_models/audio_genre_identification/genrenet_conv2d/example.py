# Copyright © 2026 Analog Devices, Inc. All Rights Reserved. This software is proprietary and confidential to Analog Devices, Inc. and its licensors.

import os
import sys
import warnings
from collections import Counter
import numpy as np
import torch
from librosa.core import load
from librosa.feature import melspectrogram
from sklearn.preprocessing import LabelEncoder
from data.model.model import genreNet

warnings.filterwarnings("ignore")

def get_config():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    return {
        "GENRES": [
            'blues', 'classical', 'country', 'disco', 'hiphop',
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ],
        "BASE_DIR": base_dir,
        "MODELPATH": os.path.join(base_dir, "data", "model", "net.pt"),
        "DATAPATH": "../data/",
        "RAW_DATAPATH": "../utils/raw_data.pkl",
        "SET_DATAPATH": "../utils/set.pkl",
    }

def load_model(map_location='cpu'):
    cfg = get_config()
    model_path = cfg["MODELPATH"]
    net = genreNet()
    state = torch.load(model_path, map_location=map_location)
    net.load_state_dict(state)
    net.eval()
    return net

def load_audio(audio_path):
    audio_path = os.path.abspath(audio_path)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    y, sr = load(audio_path, mono=True, sr=22050)
    return y, sr

def get_audio_chunks(y, sr):
    S = melspectrogram(y=y, sr=sr).T
    S = S[:-1 * (S.shape[0] % 128)]
    num_chunk = S.shape[0] / 128
    data_chunks = np.split(S, num_chunk)
    return data_chunks

def classify_chunks(net, data_chunks, le):
    genres = []
    for data in data_chunks:
        data = torch.FloatTensor(data).view(1, 1, 128, 128)
        preds = net(data)
        pred_val, pred_index = preds.max(1)
        pred_index = pred_index.data.numpy()
        pred_val = np.exp(pred_val.data.numpy()[0])
        pred_genre = le.inverse_transform(pred_index).item()
        if pred_val >= 0.5:
            genres.append(pred_genre)
    return genres

def get_genre_distribution(genres):
    counts = dict(Counter(genres))
    total = float(sum(counts.values()))
    distribution = sorted(
        [(genre, count / total * 100) for genre, count in counts.items()],
        key=lambda x: x[1],
        reverse=True
    )
    return distribution

def print_results(distribution):
    for genre, percentage in distribution:
        print("%10s: \t%.2f\t%%" % (genre, percentage))

def main(argv):
    cfg = get_config() 
    base_dir = cfg["BASE_DIR"]
    default_audio = os.path.join(base_dir, "data", "input", "test.wav")
    audio_path = argv[0] if argv else default_audio
    le = LabelEncoder().fit(cfg["GENRES"])
    net = load_model()       
    net.eval()

    with torch.no_grad():
        y, sr = load_audio(audio_path)           
        data_chunks = get_audio_chunks(y, sr)       
        genres = classify_chunks(net, data_chunks, le)
        distribution = get_genre_distribution(genres)

    print_results(distribution)

if __name__ == "__main__":
    main(sys.argv[1:])
