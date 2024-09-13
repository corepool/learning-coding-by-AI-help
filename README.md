# learning-coding-by-AI-help
# After i learned several coursed by https://learn.deeplearning.ai/. I start to write some code with the ai chatbot help. here is some.
import cv2
import numpy as np

# Load the video file
video = cv2.VideoCapture(r'your vide.mp4')

# Get the frame rate of the video
fps = video.get(cv2.CAP_PROP_FPS)

# Initialize previous coordinates, time, and speed flag
prev_x, prev_y = None, None
prev_time = None
speed_calculated = False

# Real-world tennis court length and its length in video (in pixels)
court_length_meters = 23.77
court_length_pixels = 800  # replace this with your actual measurement

# Calculate pixels to meters conversion factor
pixels_to_meters = court_length_meters / court_length_pixels

# Define the speed threshold in meters per second to identify when the ball leaves the racket
speed_threshold = 10  # Adjust this value based on your observations

while True:
    ret, frame = video.read()
    
    if not ret:
        break

    # Convert the frame to grayscale (optional, depends on your detection strategy)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Simple static thresholding as a placeholder for proper object detection
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours (outlines of objects) in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Let's assume the largest contour in the frame is the ball 
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Get current time in seconds
        current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds
        
        if prev_x is not None and prev_y is not None and prev_time is not None:
            # Calculate distance (Euclidean distance in pixels)
            distance_pixels = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            
            # Convert distance to meters
            distance_meters = distance_pixels * pixels_to_meters
            
            # Calculate elapsed time
            elapsed_time = current_time - prev_time
            
            if elapsed_time > 0:  # Avoid division by zero
                # Calculate speed (meters per second)
                speed_mps = distance_meters / elapsed_time
                
                if speed_mps > speed_threshold and not speed_calculated:
                    # Convert speed to kilometers per hour
                    speed_kmph = speed_mps * 3.6
                    print(f'Speed at racket leaving: {speed_kmph:.2f} km/h')
                    speed_calculated = True
        
        # Update previous coordinates and time
        prev_x, prev_y = x, y
        prev_time = current_time
    
    # Show the frame with the detected object
    cv2.imshow('Frame', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
