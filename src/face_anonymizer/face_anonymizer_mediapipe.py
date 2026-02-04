"""Face Anonymizer using mediapipe

This module anonymizes faces in a video stream by blurring them.

Requirements:
    - opencv-python
    - mediapipe
"""

import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def download_model_if_needed(model_path, model_url):
    """Download model file if it doesn't exist."""
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully!")


def main():
    """Main function to anonymize faces in video stream using MediaPipe."""
    # open camera
    video_capture = cv2.VideoCapture(0)

    # construct absolute path to model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_path = os.path.join(project_root, 'models', 'face_landmarker.task')
    base_url = "https://storage.googleapis.com/mediapipe-models"
    model_url = f"{base_url}/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    download_model_if_needed(model_path, model_url)

    # initialize face detector
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        num_faces=3)
    face_detector = vision.FaceLandmarker.create_from_options(options)

    # read frame
    frame_timestamp_ms = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # detect face
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = face_detector.detect_for_video(mp_image, frame_timestamp_ms)

        # anonymize face (blur)
        if detection_result.face_landmarks:
            frame_height, frame_width = frame.shape[:2]
            for face_landmarks in detection_result.face_landmarks:
                x_coords = [landmark.x * frame_width for landmark in face_landmarks]
                y_coords = [landmark.y * frame_height for landmark in face_landmarks]
                x_min = int(min(x_coords))
                y_min = int(min(y_coords))
                x_max = int(max(x_coords))
                y_max = int(max(y_coords))
                face_region = frame[y_min:y_max, x_min:x_max]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y_min:y_max, x_min:x_max] = blurred_face

        # show frame
        cv2.imshow('Frame', frame)

        # wait for key press and exit if q is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        frame_timestamp_ms += 1

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
