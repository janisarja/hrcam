import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from face_detection import detect_face, extract_roi
from processing import filter_roi, process_signal
from plot import Plot

def update_roi_video(filtered_roi, roi_canvas):
    # Convert the filtered ROI to PhotoImage and display it in the ROI canvas
    roi_rgb = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2RGB)
    roi_img = Image.fromarray(roi_rgb)
    roi_img_tk = ImageTk.PhotoImage(image=roi_img)
    roi_canvas.create_image(0, 0, image=roi_img_tk, anchor=tk.NW)
    roi_canvas.image = roi_img_tk

def update_webcam_video(frame, video_canvas):
    # Convert the frame to PhotoImage and display it in the video canvas (even when forehead not detected)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    video_canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
    video_canvas.image = img_tk

def create_plots(parent_frame, filter_settings):
    plots = []

    # Savgol Filter
    savgol_plot_frame = ttk.Frame(parent_frame)
    savgol_plot = Plot(savgol_plot_frame, title="Savitzky-Golay Filter")

    savgol_toggle_button = tk.Checkbutton(savgol_plot_frame, text="Use Savitzky-Golay Filtering", variable=filter_settings[0]['use'])
    savgol_toggle_button.grid(row=1, column=0, columnspan=2, sticky="w")
    
    savgol_window_label = tk.Label(savgol_plot_frame, text="Window Size")
    savgol_window_label.grid(row=1, column=0, sticky="e")
    savgol_window_selector = tk.Spinbox(savgol_plot_frame, from_=6, to=51, width=5, textvariable=filter_settings[0]['window'])
    savgol_window_selector.grid(row=1, column=1, sticky="w")

    savgol_polyorder_label = tk.Label(savgol_plot_frame, text="Polyorder")
    savgol_polyorder_label.grid(row=2, column=0, sticky="e")
    savgol_polyorder_selector = tk.Spinbox(savgol_plot_frame, from_=2, to=5, width=5, textvariable=filter_settings[0]['polyorder'])
    savgol_polyorder_selector.grid(row=2, column=1, sticky="w")

    savgol_plot_frame.grid(row=1, column=1, padx=5, pady=5)

    plots.append(savgol_plot)

    # Next Filter

    return plots

def setup_gui(root):
    root.title("HRCam")

    # Create three main sections using frames
    left_frame = ttk.Frame(root)
    left_frame.grid(row=0, column=0, sticky="nsew")

    center_frame = ttk.Frame(root)
    center_frame.grid(row=0, column=1, sticky="nsew")

    right_frame = ttk.Frame(root)
    right_frame.grid(row=0, column=2, sticky="nsew")

    # Adjust column and row weights for layout control
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=3)
    root.grid_columnconfigure(2, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # Video feeds on the left side
    video_canvas = tk.Canvas(left_frame, width=640, height=480)
    video_canvas.pack()

    roi_canvas = tk.Canvas(left_frame, width=320, height=240)
    roi_canvas.pack()

    # Section in the middle for red channel signal plots
    plot_grid_frame = ttk.Frame(center_frame)
    plot_grid_frame.pack(fill=tk.BOTH, expand=True)

    # Right section for heart rate display
    heart_rate_frame = ttk.Frame(right_frame, relief=tk.SUNKEN)
    heart_rate_frame.pack(fill=tk.BOTH, expand=True)

    return video_canvas, roi_canvas, plot_grid_frame, heart_rate_frame

def update_gui(root, cap, video_canvas, roi_canvas, plots, frame_count, x_data, y_data_processed, y_data, filter_settings):
    ret, frame = cap.read()
    if not ret:
        root.quit()
    
    grey_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect_face(grey_channel)

    if face:
        roi = extract_roi(frame, grey_channel, face)
        filtered_roi = filter_roi(roi)
        average_intensity = np.mean(filtered_roi)

        # Update the signal data
        frame_count += 1
        x_data.append(frame_count)
        y_data.append(average_intensity)

        process_signal(y_data, y_data_processed, filter_settings)

        # Update all plots
        for i in range(len(plots)):
            plots[i].update(y_data_processed[i][-1])

        update_roi_video(filtered_roi, roi_canvas)

    update_webcam_video(frame, video_canvas)

    root.after(10, update_gui, root, cap, video_canvas, roi_canvas, plots, frame_count, x_data, y_data_processed, y_data, filter_settings)
