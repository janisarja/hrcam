import cv2
import tkinter as tk
import time
from visualization import setup_gui, update_gui, create_plots

def main():
    root = tk.Tk()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = 9

    x_data = []
    y_data = {
        'raw': [],
        'bandpass': [],
        'savgol': [],
        'bpm': []
    }

    # Default values for each filter parameters.
    median_blur_settings = {
        'use': tk.BooleanVar(value=True)}
    bandpass_settings = {
        'use': tk.BooleanVar(value=True), 
        'lowcut': tk.StringVar(value=0.7), 
        'highcut': tk.StringVar(value=3.0), 
        'order': tk.IntVar(value=6)}
    savgol_settings = {
        'use': tk.BooleanVar(value=True), 
        'window': tk.IntVar(value=31), 
        'polyorder': tk.IntVar(value=3)}

    filter_settings = {
        'blur': median_blur_settings,
        'bandpass': bandpass_settings,
        'savgol': savgol_settings
    }

    video_canvas, roi_canvas, plot_grid_frame, heart_rate_label, hr_plot = setup_gui(root, filter_settings)

    plot_list = create_plots(plot_grid_frame, filter_settings)
    plots = {
        'raw': plot_list[0],
        'bandpass': plot_list[1],
        'savgol': plot_list[2],
        'bpm': hr_plot
    }

    start_time = time.time()

    update_gui(root, cap, video_canvas, roi_canvas, plots, start_time, x_data, y_data, filter_settings, heart_rate_label, fps)

    root.mainloop()

    cap.release()

if __name__ == '__main__':
    main()
