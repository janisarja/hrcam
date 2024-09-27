import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageTk

frame_count = 0

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Set up the Tkinter window
root = tk.Tk()
root.title("HRCam")

# Create a canvas for the video feed
video_canvas = tk.Canvas(root, width=640, height=480)
video_canvas.pack(side=tk.LEFT)

# Create a canvas for the region of interest (forehead)
roi_canvas = tk.Canvas(root, width=320, height=240)
roi_canvas.pack(side=tk.LEFT)

# Set up for plotting the signal
fig, ax = plt.subplots(figsize=(5, 4))
x_data = []
y_data = []
line, = ax.plot(x_data, y_data)

# Set the axes limits and labels
ax.set_xlim(0, 100)
ax.set_ylim(0, 255)
ax.set_xlabel('Frame Count')
ax.set_ylabel('Average Red Intensity')

# Create a canvas for the plot
plot_canvas = FigureCanvasTkAgg(fig, master=root)
plot_canvas_widget = plot_canvas.get_tk_widget()
plot_canvas_widget.pack(side=tk.RIGHT)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

def update_plot():
    global x_data, y_data
    if len(x_data) > 0:
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.set_xlim(max(0, frame_count - 100), frame_count)
        # Update y-axis limits every 20 frames to fit all values
        if len(x_data) % 20 == 0:
            ax.set_ylim(min(y_data[-100:]) - 10, max(y_data[-100:]) + 10)
        plot_canvas.draw()

def update_video():
    global frame_count, x_data, y_data

    ret, frame = cap.read()
    if not ret:
        root.quit()

    # Detect faces in the frame
    gray_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_channel)

    if len(faces) > 0:
        # Only processing first detected face if multiple
        face = faces[0]
        
        # Get landmarks for the detected face
        landmarks = predictor(gray_channel, face)

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

        # Extract red channel and filter with median blurring
        red_channel = roi[:, :, 2]
        filtered_roi = cv2.medianBlur(red_channel, 5)

        # Calculate average intensity
        average_intensity = np.mean(filtered_roi)

        # Update the signal data
        frame_count += 1
        x_data.append(frame_count)
        y_data.append(average_intensity)

        # Update the plot
        update_plot()

        # Convert the filtered ROI to PhotoImage and display it in the ROI canvas
        roi_rgb = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2RGB)
        roi_img = Image.fromarray(roi_rgb)
        roi_img_tk = ImageTk.PhotoImage(image=roi_img)
        roi_canvas.create_image(0, 0, image=roi_img_tk, anchor=tk.NW)
        roi_canvas.image = roi_img_tk  # Keep a reference

    # Convert the frame to PhotoImage and display it in the video canvas (even when forehead not detected)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    video_canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
    video_canvas.image = img_tk  # Keep a reference

    # Schedule the next frame update
    video_canvas.after(10, update_video)

update_video()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
