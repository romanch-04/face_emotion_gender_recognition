import cv2
from deepface import DeepFace
import os

# Hide TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Variables for optimization
frame_count = 0
skip_frames = 5  # Only analyze every 5th frame
last_results = [] # Store results to show during skipped frames

print("System optimized for speed. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only run the heavy AI every 'skip_frames'
    if frame_count % skip_frames == 0:
        try:
            # Resize frame to be smaller for FASTER processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Analyze the small frame
            last_results = DeepFace.analyze(
                small_frame, 
                actions=['emotion', 'gender'], 
                enforce_detection=False,
                detector_backend='opencv' # Fastest backend
            )
        except Exception:
            last_results = []

    # Draw the results (even on frames we skipped)
    if last_results:
        for res in last_results:
            # Multiply coordinates by 2 because we analyzed a half-size frame
            x = res['region']['x'] * 2
            y = res['region']['y'] * 2
            w = res['region']['w'] * 2
            h = res['region']['h'] * 2
            
            emotion = res['dominant_emotion']
            gender = res['dominant_gender']

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{gender}, {emotion}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Fast Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()