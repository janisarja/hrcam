import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from face_detection import detect_face, extract_roi
from processing import filter_roi, process_signal, calculate_bpm
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

    # 1. Unfiltered Signal
    raw_plot_frame = ttk.Frame(parent_frame)
    raw_plot = Plot(raw_plot_frame, title=" 1. Unfiltered Signal")

    raw_plot_frame.grid(row=0, column=0, padx=5, pady=5)

    plots.append(raw_plot)

    # 2. Savgol Filter
    savgol_plot_frame = ttk.Frame(parent_frame)
    savgol_plot = Plot(savgol_plot_frame, title="2. Savitzky-Golay Filter")

    savgol_toggle_button = tk.Checkbutton(savgol_plot_frame, text="Use Savitzky-Golay Filtering", variable=filter_settings[1]['use'])
    savgol_toggle_button.grid(row=1, column=0, columnspan=2, sticky="w")
    
    savgol_window_label = tk.Label(savgol_plot_frame, text="Window Size")
    savgol_window_label.grid(row=1, column=0, sticky="e")
    savgol_window_selector = tk.Spinbox(savgol_plot_frame, from_=6, to=51, width=5, textvariable=filter_settings[1]['window'])
    savgol_window_selector.grid(row=1, column=1, sticky="w")

    savgol_polyorder_label = tk.Label(savgol_plot_frame, text="Polyorder")
    savgol_polyorder_label.grid(row=2, column=0, sticky="e")
    savgol_polyorder_selector = tk.Spinbox(savgol_plot_frame, from_=2, to=5, width=5, textvariable=filter_settings[1]['polyorder'])
    savgol_polyorder_selector.grid(row=2, column=1, sticky="w")

    savgol_plot_frame.grid(row=1, column=0, padx=5, pady=5)

    plots.append(savgol_plot)

    # Next Filter

    return plots

def setup_gui(root, filter_settings):
    root.title("HRCam")
    root.wm_state("zoomed")

    # Create three main sections using frames
    left_frame = ttk.Frame(root)
    left_frame.grid(row=0, column=0, sticky="nsew")

    center_frame = ttk.Frame(root)
    center_frame.grid(row=0, column=1, sticky="nsew")

    right_frame = ttk.Frame(root)
    right_frame.grid(row=0, column=2, sticky="nsew")

    # Set minimum width for the left side frame (adjust based on video size)
    min_left_width = 640  # Set this to the width of the full-size video
    root.grid_columnconfigure(0, weight=0, minsize=min_left_width)  # No scaling on the left

    # Set the center and right sections to be resizable
    root.grid_columnconfigure(1, weight=3)  # Center plot grid can scale
    root.grid_columnconfigure(2, weight=1)  # Right section scales proportionally
    root.grid_rowconfigure(0, weight=1)

    # Video feeds on the left side
    video_canvas = tk.Canvas(left_frame, width=640, height=480)  # Full-size video
    video_canvas.pack()

    blur_toggle_button = tk.Checkbutton(left_frame, text="Use Median Blurring", variable=filter_settings[0]['use'])
    blur_toggle_button.pack()

    roi_canvas = tk.Canvas(left_frame, width=320, height=240)
    roi_canvas.pack()

    # Section in the middle for red channel signal plots with scrollable functionality
    canvas_frame = tk.Frame(center_frame)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    # Create a canvas for the middle section plots
    plot_canvas = tk.Canvas(canvas_frame)
    plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add a vertical scrollbar linked to the canvas
    scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=plot_canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configure the canvas to be scrollable
    plot_canvas.configure(yscrollcommand=scrollbar.set)
    plot_canvas.bind('<Configure>', lambda e: plot_canvas.configure(scrollregion=plot_canvas.bbox("all")))

    # Make scrollable with mouse wheel
    def _on_mouse_wheel(event):
        plot_canvas.yview_scroll(-1 * int((event.delta / 120)), "units")  # For Windows/macOS
    plot_canvas.bind_all("<MouseWheel>", _on_mouse_wheel)
    plot_canvas.bind_all("<Button-4>", lambda event: plot_canvas.yview_scroll(-1, "units"))  # For Linux
    plot_canvas.bind_all("<Button-5>", lambda event: plot_canvas.yview_scroll(1, "units"))   # For Linux

    # Create a frame within the canvas to hold the plots
    plot_grid_frame = ttk.Frame(plot_canvas)
    plot_canvas.create_window((0, 0), window=plot_grid_frame, anchor="nw")

    # Right section for heart rate display
    heart_rate_frame = ttk.Frame(right_frame, relief=tk.SUNKEN)
    heart_rate_frame.pack(fill=tk.BOTH, expand=True)

    heart_rate_label = tk.Label(heart_rate_frame, text="-- bpm", font=("Arial", 16))
    heart_rate_label.pack(pady=10)

    # Plot heart rate
    hr_plot_frame = ttk.Frame(right_frame)
    hr_plot = Plot(hr_plot_frame, title="Heart Rate", y_label="bpm")

    hr_plot_frame.pack(pady=10)

    return video_canvas, roi_canvas, plot_grid_frame, heart_rate_label, hr_plot

def update_gui(root, cap, video_canvas, roi_canvas, plots, start_time, x_data, y_data, filter_settings, heart_rate_label, hr_plot):
    ret, frame = cap.read()
    if not ret:
        root.quit()

    elapsed_time = time.time() - start_time
    
    grey_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect_face(grey_channel)

    if face:
        roi = extract_roi(frame, grey_channel, face)
        filtered_roi = filter_roi(roi, filter_settings[0]['use'].get())
        average_intensity = np.mean(filtered_roi)

        # Update the signal data
        x_data.append(elapsed_time)
        y_data[0].append(average_intensity)

        process_signal(y_data, filter_settings)

        # Update signal plots
        for i in range(len(plots)):
            plots[i].update(elapsed_time, y_data[i][-1])

        bpm = calculate_bpm(x_data, y_data, filter_settings)

        if bpm is not None:
            heart_rate_label.config(text=f"{int(bpm)} bpm")

        hr_plot.update(elapsed_time, bpm)

        update_roi_video(filtered_roi, roi_canvas)

    update_webcam_video(frame, video_canvas)

    root.after(10, update_gui, root, cap, video_canvas, roi_canvas, plots, start_time, x_data, y_data, filter_settings, heart_rate_label, hr_plot)
