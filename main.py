import cv2
import tkinter as tk
from visualization import setup_gui, update_gui

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

if __name__ == '__main__':
    main()
