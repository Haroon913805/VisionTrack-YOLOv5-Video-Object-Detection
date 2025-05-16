import torch
import cv2
import os

# Load pretrained YOLOv5 model from torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_on_video(video_path):
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Unable to open video file.")
        return

    # Define window name
    window_name = 'YOLOv5 Video Detection'

    # Create a resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set window size (width x height)
    cv2.resizeWindow(window_name, 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(img_rgb)

        # Render results (draw boxes)
        results.render()  # updates results.imgs with boxes and labels

        # Convert back to BGR for OpenCV display
        img_result = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
        

        # Show frame with detections
        cv2.imshow(window_name, img_result)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Change this to your video path
    input_path = r'./dataa/road.mp4'  # Example video path

    # Check if the input path is a valid video file
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detect_on_video(input_path)
    else:
        print("Please provide a valid video file (.mp4, .avi, .mov, .mkv).")
