import cv2
import tkinter as tk
from visualization import setup_gui, update_gui, create_plots

def main():
    root = tk.Tk()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    x_data = []
    y_data_processed = [[]]
    y_data = []

    video_canvas, roi_canvas, plot_grid_frame, heart_rate_frame = setup_gui(root)

    # Default values for each filter parameters.
    filter_settings = []
    savgol_settings = {'use': tk.BooleanVar(value=True), 'window': tk.IntVar(value=31), 'polyorder': tk.IntVar(value=3)}
    filter_settings.append(savgol_settings)
    

    plots = create_plots(plot_grid_frame, filter_settings)

    update_gui(root, cap, video_canvas, roi_canvas, plots, frame_count, x_data, y_data_processed, y_data, filter_settings)

    root.mainloop()

    cap.release()

if __name__ == '__main__':
    main()
