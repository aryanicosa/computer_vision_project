"""
Posture detection system using MediaPipe Pose Landmarker.

This module monitors user posture by analyzing neck inclination angle
using the ear and shoulder landmarks. It provides real-time feedback
on posture quality (GOOD, WARN, BAD POSTURE).

Requirements:
    - mediapipe==0.10.32
    - opencv-python
    - numpy
"""

import time

import cv2
import mediapipe as mp
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = '../../models/pose_landmarker_lite.task' # Ensure this file is in your folder!
GOOD_POSTURE_THRESHOLD = 15  # Degrees
BAD_POSTURE_THRESHOLD = 35   # Degrees

# --- MODERN API SETUP ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize the Landmarker
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO # Optimized for webcam feeds
)

def calculate_neck_inclination(ear_landmark, shoulder_landmark):
    """
    Calculates angle of neck (ear-shoulder) vs Vertical line.

    Args:
        ear_landmark: MediaPipe landmark object for the ear position
        shoulder_landmark: MediaPipe landmark object for the shoulder position

    Returns:
        float: The neck inclination angle in degrees
    """
    # Create a point directly above the shoulder
    vertical_point = [shoulder_landmark.x, shoulder_landmark.y - 0.5]

    # Calculate angle using arctan2
    radians = np.arctan2(
        vertical_point[1] - shoulder_landmark.y,
        vertical_point[0] - shoulder_landmark.x
    ) - np.arctan2(
        ear_landmark.y - shoulder_landmark.y,
        ear_landmark.x - shoulder_landmark.x
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def draw_landmarks(image, pose_landmarks):
    """
    Custom drawing function to visualize pose landmarks and connections.

    Args:
        image: The image to draw on (numpy array)
        pose_landmarks: List of MediaPipe landmark objects
    """
    img_height, img_width, _ = image.shape

    # Draw connections (lines)
    connections = PoseLandmarksConnections.POSE_LANDMARKS
    for connection in connections:
        start_idx = connection.start
        end_idx = connection.end

        # Check if points exist in detection
        if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
            start_point = (
                int(pose_landmarks[start_idx].x * img_width),
                int(pose_landmarks[start_idx].y * img_height)
            )
            end_point = (
                int(pose_landmarks[end_idx].x * img_width),
                int(pose_landmarks[end_idx].y * img_height)
            )
            cv2.line(image, start_point, end_point, (245, 66, 230), 2)

    # Draw points (circles)
    for landmark in pose_landmarks:
        cx, cy = int(landmark.x * img_width), int(landmark.y * img_height)
        cv2.circle(image, (cx, cy), 5, (245, 117, 66), -1)

# --- MAIN LOOP ---
# Create the landmarker instance
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    # We need a start time to calculate timestamps for the AI
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Prepare Image for MediaPipe
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 2. Calculate Timestamp (Required for VIDEO mode)
        # MediaPipe expects microseconds (us)
        timestamp_ms = int((time.time() - start_time) * 1000)

        # 3. Detect
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 4. Process Results
        if detection_result.pose_landmarks:
            # Get the first person detected
            landmarks = detection_result.pose_landmarks[0]

            # Indices: Left Ear (7), Left Shoulder (11)
            ear = landmarks[7]
            shoulder = landmarks[11]

            # Calculate Angle
            neck_inclination = calculate_neck_inclination(ear, shoulder)

            # Determine Status
            if neck_inclination < GOOD_POSTURE_THRESHOLD:
                posture_status = "GOOD"
                status_color = (0, 255, 0)  # Green
            elif neck_inclination < BAD_POSTURE_THRESHOLD:
                posture_status = "WARN"
                status_color = (0, 255, 255)  # Yellow
            else:
                posture_status = "BAD POSTURE"
                status_color = (0, 0, 255)  # Red

            # Draw Visuals
            frame_height, frame_width, _ = frame.shape

            # Draw Status Text
            cv2.putText(
                frame,
                f"{int(neck_inclination)} deg",
                (int(shoulder.x * frame_width), int(shoulder.y * frame_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                posture_status,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                status_color,
                3
            )

            # Draw Skeleton
            draw_landmarks(frame, landmarks)

        cv2.imshow('Ergonomic Guard (Modern API)', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
