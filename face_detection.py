import dlib
import cv2

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

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