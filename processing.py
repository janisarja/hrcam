import cv2
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
import numpy as np

def filter_roi(roi, use_blurring):
    # Extract red channel and filter with median blurring and min-max normalization
    red_channel = roi[:, :, 2]
    if use_blurring:
        red_channel = cv2.medianBlur(red_channel, 5)
    normalized_roi = cv2.normalize(red_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_roi

def savgol(y_data, window, polyorder):
    if len(y_data[1]) >= window and not (None in y_data[1][-window:]):
        smoothed_signal = savgol_filter(y_data[1][-window:], window_length=window, polyorder=polyorder)
        y_data[2].append(smoothed_signal[-1])
    else:
        y_data[2].append(None)

def bandpass(y_data, lowcut, highcut, fs, order):
    if len(y_data[0]) > max(order + 1, 39):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y_data[1] = filtfilt(b, a, y_data[0])
    else:
        y_data[1].append(None)

def process_signal(y_data, filter_settings):
    fs = 30
    # Applpy bandpass filtering
    if filter_settings[1]['use'].get():
        lowcut = float(filter_settings[1]['lowcut'].get())
        highcut = float(filter_settings[1]['highcut'].get())
        order = filter_settings[1]['order'].get()
        bandpass(y_data, lowcut, highcut, fs, order)
    else:
        y_data[1].append(None)
    
    # Smooth signal with Savitzky-Golay filtering
    if filter_settings[2]['use'].get():
        window = filter_settings[2]['window'].get()
        polyorder = filter_settings[2]['polyorder'].get()
        savgol(y_data, window, polyorder)
    else:
        y_data[2].append(None)

def calculate_bpm(x_data, y_data, filter_settings):
    most_filtered = 0
    # Check which level of filtering is the latest one that is toggled on.
    for i in range(len(filter_settings)):
        if filter_settings[i]['use'].get():
            most_filtered = i

    if len(y_data[most_filtered]) > 30:  # Ensure enough data for peak detection
        peaks, _ = find_peaks(y_data[most_filtered][-30:], distance=3)  # Adjust 'distance' based on expected heart rate

        if len(peaks) > 1:
            # Calculate time difference between peaks (in seconds)
            peak_times = [x_data[p] for p in peaks]
            peak_intervals = np.diff(peak_times)
            if len(peak_intervals) > 0:
                avg_peak_interval = np.mean(peak_intervals)  # Average time between peaks
                bpm = 60 / avg_peak_interval  # Convert to beats per minute
                return bpm
    return None
