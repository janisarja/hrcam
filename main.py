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

    x_data = []
    y_data = [[], [], []]

    # Default values for each filter parameters.
    filter_settings = []

    median_blur_settings = {'use': tk.BooleanVar(value=True)}
    bandpass_settings = {
        'use': tk.BooleanVar(value=True), 
        'lowcut': tk.StringVar(value=0.7), 
        'highcut': tk.StringVar(value=3.0), 
        'order': tk.IntVar(value=6)}
    savgol_settings = {
        'use': tk.BooleanVar(value=True), 
        'window': tk.IntVar(value=31), 
        'polyorder': tk.IntVar(value=3)}

    filter_settings.append(median_blur_settings)
    filter_settings.append(bandpass_settings)
    filter_settings.append(savgol_settings)

    video_canvas, roi_canvas, plot_grid_frame, heart_rate_label, hr_plot = setup_gui(root, filter_settings)

    plots = create_plots(plot_grid_frame, filter_settings)

    start_time = time.time()

    update_gui(root, cap, video_canvas, roi_canvas, plots, start_time, x_data, y_data, filter_settings, heart_rate_label, hr_plot)

    root.mainloop()

    cap.release()

if __name__ == '__main__':
    main()
