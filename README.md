# hrcam - Webcam Heart Rate Detection

This project captures webcam video and estimates heart rate based on changes in the color intensity of the forehead skin. It utilizes OpenCV for video processing and dlib for facial landmark detection, along with Matplotlib to visualize the heart rate signal in real time.

## How It Works

Measuring heart rate from video involves detecting subtle changes in skin color associated with blood flow. When the heart beats, blood volume in the skin increases, leading to a slight change in color intensity, primarily in the red channel of the image.

### Process Overview:
1. **Webcam Capture**: The program uses OpenCV to access the webcam feed.
2. **Face Detection**: Dlib's facial landmark detector identifies facial features and helps isolate the forehead region, where changes in color can be observed.
3. **Region of Interest (ROI)**: The area of the forehead is extracted from the video frames for further analysis.
4. **Signal Processing**: The average intensity of the red channel is computed over time, creating a signal that reflects the pulsations corresponding to the heartbeat.
5. **Visualization**: The detected heart rate signal is plotted in real time alongside the video feed.

## Dependencies

This project requires the following Python libraries:

- **OpenCV**: For webcam video capture and image processing.
- **dlib**: For facial landmark detection and facial feature extraction.
- **NumPy**: For efficient numerical operations.
- **Matplotlib**: For live plotting of the heart rate signal.
- **Pillow**: For image handling and converting formats.
- **Tkinter**: For the GUI (Graphical User Interface) used to display the webcam feed and plot.

### Installing Dependencies

You can install the required dependencies using `pip`:

```bash
pip install opencv-python-headless dlib numpy matplotlib pillow
