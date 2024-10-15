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

    frame_count = 0
    x_data = []
    y_data = [[], []]

    # Default values for each filter parameters.
    filter_settings = []

    median_blur_settings = {'use': tk.BooleanVar(value=True)}
    savgol_settings = {'use': tk.BooleanVar(value=True), 'window': tk.IntVar(value=31), 'polyorder': tk.IntVar(value=3)}

    filter_settings.append(median_blur_settings)
    filter_settings.append(savgol_settings)

    video_canvas, roi_canvas, plot_grid_frame, heart_rate_frame, heart_rate_label = setup_gui(root, filter_settings)

    plots = create_plots(plot_grid_frame, filter_settings)

    start_time = time.time()

    update_gui(root, cap, video_canvas, roi_canvas, plots, start_time, x_data, y_data, filter_settings, heart_rate_label)

    root.mainloop()

    cap.release()

if __name__ == '__main__':
    main()
