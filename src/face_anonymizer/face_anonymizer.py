"""
Face Anonymizer

This module anonymizes faces in a video stream by blurring them.

Requirements:
    - opencv-python
"""

import cv2


def main():
    """Main function to anonymize faces in video stream."""
    # open camera
    video_capture = cv2.VideoCapture(0)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # read frame
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # detect face
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale reduce computational complexity and just focus on intensity variation
        detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        # anonymize face (blur)
        for (x_coord, y_coord, width, height) in detected_faces:
            face_region = frame[y_coord:y_coord+height, x_coord:x_coord+width]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y_coord:y_coord+height, x_coord:x_coord+width] = blurred_face
        
        # show frame
        cv2.imshow('Frame', frame)
        
        # wait for key press and exit if q is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()