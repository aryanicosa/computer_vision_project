import os
import cv2
import mediapipe as mp
import numpy as np
import time

# --- CONFIGURATION ---
# Construct absolute path to model file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
MODEL_PATH = os.path.join(project_root, 'models', 'pose_landmarker_lite.task')

# --- SETUP ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create options
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO 
)

# Custom Drawing Function (Since new API is different)
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses (usually just 1 person)
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the points
        for landmark in pose_landmarks:
            # Convert normalized coordinates (0-1) to pixel coordinates
            h, w, _ = annotated_image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)

    return annotated_image

# --- MAIN LOOP ---
# The 'with' statement keeps the model open safely
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Convert to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Convert to MediaPipe Image Object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 3. Calculate Timestamp (Required for VIDEO mode)
        timestamp_ms = int((time.time() - start_time) * 1000)

        # 4. Detect Pose
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 5. Visualize
        # Draw landmarks on the frame
        output_image = draw_landmarks_on_image(frame, detection_result)
        
        # 6. Extract Data Example
        if detection_result.pose_landmarks:
            # Get the Nose (Index 0) of the first person
            nose = detection_result.pose_landmarks[0][0] 
            cv2.putText(output_image, f"Nose Y: {nose.y:.2f}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Modern Pose Estimation', output_image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()