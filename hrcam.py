import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

forehead_detected = False

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]  # Tracking only the first detected face
        
        # Get facial landmarks
        landmarks = predictor(gray, face)

        face_height = face.bottom() - face.top()

        left_upper_eyebrow = landmarks.part(19).y
        right_upper_eyebrow = landmarks.part(24).y
        left_outer_eybrow = landmarks.part(17).x
        right_outer_eybrow = landmarks.part(26).x

        if left_upper_eyebrow < right_upper_eyebrow:
            top_eyebrow = left_upper_eyebrow
        else:
            top_eyebrow = right_upper_eyebrow

        # Define a region of interest for the forehead area
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

        # Mark forehead detection
        if not forehead_detected:
            forehead_detected = True
            print("Forehead detected.")

        # Draw a rectangle around the ROI
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (255, 0, 0), 2)
    
    else:
        # No faces detected
        if forehead_detected:
            forehead_detected = False
            print("Forehead lost.")

    # Extract red channel from the region of interest
    r, g, b = cv2.split(roi)
    zero_channel = np.zeros_like(b)
    red_roi = cv2.merge([zero_channel, zero_channel, r])

    cv2.imshow('Webcam', frame)
    cv2.imshow('ROI', red_roi)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
