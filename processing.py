import cv2

def filter_roi(roi):
    # Extract red channel and filter with median blurring
    red_channel = roi[:, :, 2]
    filtered_roi = cv2.medianBlur(red_channel, 5)
    return filtered_roi
