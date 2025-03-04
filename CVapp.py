import cv2
import torch
import numpy as np
import os
import json
from datetime import datetime
import warnings
import streamlit as st
import time
import streamlit as st
from PIL import Image  # You'll need Pillow library: pip install Pillow

# Suppress all warnings
warnings.filterwarnings("ignore")

# Replace with your RTSP stream URL
rtsp_url = "rtsp://ailab:xxxxx@192.168.1.64:554/Streaming/Channels/1"
#rtsp_url = "rtsp://tapo123:xxxx@192.168.1.2:554/stream2"


# Replace with the path to your YOLOv5 model
model_path = "yolov5s.pt" # or your model name

# Load the YOLOv5 model

log_dir = "log"
frame_count = 0
diff_sum = 0
prev_frame = None
frame_diff_thresh=500000
people_count=0
prev_people_count=0
last_frame_time = 0  # Initialize timestamp for FPS calculation
fps_target=5
similarity_threshold = 60
ssim_score = 0
detect = 0

from skimage.metrics import structural_similarity as ssim

def compare_frames(frame1, frame2):
    """Compares two frames using MSE and SSIM."""

    # Ensure frames are grayscale for MSE and SSIM
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate MSE
    mse = np.mean((gray1 - gray2) ** 2)

    # Calculate SSIM
    score, diff = ssim(gray1, gray2, full=True)
    score = score * 100  # Convert to percentage

    return mse, score




def detect_people(frame):
    results = model(frame)
    detections = results.pandas().xyxy[0]
    people_count = len(detections[detections['name'] == 'person'])
    return detections, people_count



st.title("Adani AI Video Analytics")
# Load your image
image = Image.open("download.jpeg") # Replace with your image path

with st.spinner("Initializing the application..."): 
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.conf = 0.5  # Confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    
placeholder = st.empty()
# Display the image

with placeholder.container():
    col1, col2 = st.columns(2)  #Two columns for better layout

    col1.image(image, channels="RGB",caption="Digital Eyes")

    with col2:
        button1_clicked = st.subheader("Available Cameras")
        button1_clicked = st.button("GP's Home")
        button2_clicked = st.button("GP's Desk")
        button2_clicked = st.button("GP's Office")
 
#placeholder.image(image, channels="RGB",caption="Digital Eyes")
# Display a banner/header

if button2_clicked:
   
    placeholder.write("Connecting ...")
    progress_bar = placeholder.progress(0)
    for i in range(2):
        time.sleep(1)  # Simulate a long process
        progress_bar.progress((i + 1) / 10)

    try:
        # Open the RTSP stream
        cap = cv2.VideoCapture(rtsp_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps=int(fps)
        st.write(f"FPS deteted from Camrea: {fps}")
        st.write(f"FPS Revised for Analytics: {fps_target}")
        st.write("Events Log")        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cv2.getTickCount() / cv2.getTickFrequency()  #Get current time in seconds
            time_since_last_frame = current_time - last_frame_time
            # Check if enough time has passed for the target FPS
            if time_since_last_frame >= 1.0 / fps_target:
                last_frame_time = current_time
                frame_count +=1
                detections, people_count = detect_people(frame)
                    
                # Draw bounding boxes and display count
                for index, row in detections.iterrows():
                    if row['name'] == 'person':
                        x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']].values)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add text to display people count
                cv2.putText(frame, f"People: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display the resulting frame
                #cv2.imshow('RTSP Stream with People Detection', frame)
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder.image(frame,channels="BGR",caption="GP's Office Camera")
                

                if  prev_frame is not None :
                    mse, ssim_score = compare_frames(frame, prev_frame)
                    #print(people_count,prev_people_count,ssim_score)

                # Save detection data
                # and people_count != prev_people_count 
                # Save detection data and image # frame_count % 150 == 0 or # and frame_count % 15 == 0  
                if  (prev_frame is None and people_count > 0) or \
                    (detect == 0 and people_count > 0) or \
                    (people_count > 0 and ssim_score < similarity_threshold) :
                    detect=1
                    prev_frame=frame
                    prev_people_count = people_count
                    timestamp = datetime.now().isoformat()
                    image_filename = os.path.join(log_dir, f"{timestamp}.jpg")
                    cv2.imwrite(image_filename, frame)

                    log_entry = {
                        "timestamp": timestamp,
                        "frame count" : frame_count,
                        "people_count": people_count,
                        "Previous people_count": prev_people_count,
                        "ssim_score" : ssim_score,
                        "image_path": image_filename
                
                    }

                    log_filename = os.path.join(log_dir, "detection_log.json")
                    with open(log_filename, 'a') as f:
                        json.dump(log_entry, f)
                        f.write('\n') # Add a newline to separate entries
                        st.write(log_entry)
                        st.write('\n')
                        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()