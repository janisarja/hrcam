import cv2
import numpy as np

# Open a connection to the webcam (0 usually refers to the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    r, g, b = cv2.split(frame)

    zero_channel = np.zeros_like(b)

    # r = cv2.GaussianBlur(r, (15, 15), 0)

    red_frame = cv2.merge([zero_channel, zero_channel, r])

    # Display the resulting frame
    cv2.imshow('Webcam', frame)
    cv2.imshow('Red Channel', red_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
