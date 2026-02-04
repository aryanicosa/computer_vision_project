import cv2
import numpy as np

# --- CONFIGURATION (PUT YOUR NUMBERS HERE) ---
lower_color = np.array([22, 35, 114]) 
upper_color = np.array([147, 106, 246])
# ---------------------------------------------

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Create the Mask (Black & White)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Optional: Clean up noise (removes tiny white dots)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 3. Find Contours (Outlines of the white blobs)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Find the largest blob (If any exist)
        if len(contours) > 0:
            # Get the biggest contour (max area)
            c = max(contours, key=cv2.contourArea)
            
            # Calculate the Center (Moments)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 5. Draw on the frame
                # Draw a green circle at the center
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1) 
                # Draw the contour outline
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                
                # Print coordinates to console (Simulating sending data to robot)
                print(f"Target Found at: X={cx}, Y={cy}")
        
        else:
            print("Target Lost")

        # Show the result
        cv2.imshow("Object Tracker", frame)
        # cv2.imshow("Mask", mask) # Uncomment this if you want to see the mask too

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()