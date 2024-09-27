import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

def update_plot(line, plot_canvas, ax, frame_count, x_data, y_data):
    if len(x_data) > 0:
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.set_xlim(max(0, frame_count - 100), frame_count)
        # Update y-axis limits every 20 frames to fit all values
        if len(x_data) % 20 == 0:
            ax.set_ylim(min(y_data[-100:]) - 10, max(y_data[-100:]) + 10)
        plot_canvas.draw()

def detect_face(grey_channel):
    faces = detector(grey_channel)

    if len(faces) > 0:
        return faces[0]
    return None

def extract_roi(frame, grey_channel, face):
    landmarks = predictor(grey_channel, face)

    face_height = face.bottom() - face.top()

    left_upper_eyebrow = landmarks.part(19).y
    right_upper_eyebrow = landmarks.part(24).y
    left_outer_eybrow = landmarks.part(17).x
    right_outer_eybrow = landmarks.part(26).x

    if left_upper_eyebrow < right_upper_eyebrow:
        top_eyebrow = left_upper_eyebrow
    else:
        top_eyebrow = right_upper_eyebrow

    # Define a ROI for the forehead area
    roi_top = top_eyebrow - int(face_height * 0.3)
    roi_bottom = top_eyebrow
    roi_left = left_outer_eybrow
    roi_right = right_outer_eybrow

    # Ensure ROI coordinates are within frame boundaries
    roi_top = max(0, roi_top)
    roi_bottom = min(frame.shape[0], roi_bottom)
    roi_left = max(0, roi_left)
    roi_right = min(frame.shape[1], roi_right)

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (255, 0, 0), 2)

    return roi

def filter_roi(roi):
    # Extract red channel and filter with median blurring
    red_channel = roi[:, :, 2]
    filtered_roi = cv2.medianBlur(red_channel, 5)
    return filtered_roi

def update_webcam_video(frame, video_canvas):
    # Convert the frame to PhotoImage and display it in the video canvas (even when forehead not detected)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    video_canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
    video_canvas.image = img_tk  # Keep a reference

def update_roi_video(filtered_roi, roi_canvas):
    # Convert the filtered ROI to PhotoImage and display it in the ROI canvas
    roi_rgb = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2RGB)
    roi_img = Image.fromarray(roi_rgb)
    roi_img_tk = ImageTk.PhotoImage(image=roi_img)
    roi_canvas.create_image(0, 0, image=roi_img_tk, anchor=tk.NW)
    roi_canvas.image = roi_img_tk  # Keep a reference

def setup_gui(root, cap):
    root.title("HRCam")

    # Create a canvas for the video feed
    video_canvas = tk.Canvas(root, width=640, height=480)
    video_canvas.pack(side=tk.LEFT)

    # Create a canvas for the region of interest (forehead)
    roi_canvas = tk.Canvas(root, width=320, height=240)
    roi_canvas.pack(side=tk.LEFT)

    # Set up for plotting the signal
    fig, ax = plt.subplots(figsize=(5, 4))
    line, = ax.plot([], [])

    # Set the axes limits and labels
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 255)
    ax.set_xlabel('Frame Count')
    ax.set_ylabel('Average Red Intensity')

    # Create a canvas for the plot
    plot_canvas = FigureCanvasTkAgg(fig, master=root)
    plot_canvas_widget = plot_canvas.get_tk_widget()
    plot_canvas_widget.pack(side=tk.RIGHT)

    return ax, line, video_canvas, roi_canvas, plot_canvas

def update_gui(root, cap, ax, line, video_canvas, roi_canvas, plot_canvas, frame_count, x_data, y_data):
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

        update_plot(line, plot_canvas, ax, frame_count, x_data, y_data)

        update_roi_video(filtered_roi, roi_canvas)

    update_webcam_video(frame, video_canvas)

    root.after(10, update_gui, root, cap, ax, line, video_canvas, roi_canvas, plot_canvas, frame_count, x_data, y_data)

def main():
    root = tk.Tk()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0
    x_data = []
    y_data = []

    ax, line, video_canvas, roi_canvas, plot_canvas = setup_gui(root, cap)

    update_gui(root, cap, ax, line, video_canvas, roi_canvas, plot_canvas, frame_count, x_data, y_data)

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()    

if __name__ == '__main__':
    main()
