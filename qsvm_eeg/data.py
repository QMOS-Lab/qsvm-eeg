import pandas as pd
import numpy as np


def load_raw_data(eeg_path, bis_path):
    try:
        df_eeg = pd.read_csv(eeg_path)
        df_bis = pd.read_csv(bis_path)
        eeg = df_eeg['EEG'].interpolate('linear').values.flatten()
        bis = df_bis['BIS'].interpolate('linear').values.flatten()
        return eeg, bis
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None


def trim_zero_ends(eeg, bis, fs_eeg=128, fs_bis=1):
    bis = np.array(bis)
    eeg = np.array(eeg)

    bis_start_idx = next((i for i, val in enumerate(bis) if val != 0), None)
    bis_end_idx = next((i for i, val in enumerate(bis[::-1]) if val != 0), None)

    if bis_start_idx is None or bis_end_idx is None:
        return np.array([]), np.array([])

    bis_end_idx = len(bis) - bis_end_idx

    start_time = bis_start_idx / fs_bis
    end_time = bis_end_idx / fs_bis

    start_eeg_idx = int(start_time * fs_eeg)
    end_eeg_idx = int(end_time * fs_eeg)

    return eeg[start_eeg_idx:end_eeg_idx], bis[bis_start_idx:bis_end_idx]
