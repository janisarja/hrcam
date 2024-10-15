import cv2
from scipy.signal import savgol_filter

def filter_roi(roi):
    # Extract red channel and filter with median blurring and min-max normalization
    red_channel = roi[:, :, 2]
    blurred_roi = cv2.medianBlur(red_channel, 5)
    normalized_roi = cv2.normalize(blurred_roi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_roi

def process_signal(y_data, y_data_processed, filter_settings):
    # Smooth signal with Savitzky-Golay filtering
    if filter_settings[0]['use'].get():
        savgol_window = filter_settings[0]['window'].get()
        savgol_polyorder = filter_settings[0]['polyorder'].get()

        if len(y_data) >= savgol_window:
            smoothed_signal = savgol_filter(y_data[-savgol_window:], window_length=savgol_window, polyorder=savgol_polyorder)
            y_data_processed[0].append(smoothed_signal[-1])
        else:
            y_data_processed[0].append(y_data[-1])
    else:
        y_data_processed[0].append(None)
