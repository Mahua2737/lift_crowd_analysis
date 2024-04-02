import streamlit as st
import cv2
import torch
import time
from PIL import Image

# Load YOLO model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")


# Function to get person count from a frame
def get_person_count(frame, selected_floor, selected_lift):
    results = model(frame)
    detections = []

    if not results.pandas().xyxy[0].empty:
        for result in results.pandas().xyxy[0].itertuples():
            if result[7] == "person":
                detections.append(
                    [int(result[1]), int(result[2]), int(result[3]), int(result[4])]
                )

    return len(detections)


# Main function to create the UI
def main():
    st.title("QLess")

    
    selected_floor = st.sidebar.selectbox("Select Floor", list(range(-1, 11)), index=1)

    
    count_placeholder = st.empty()

    
    icon_placeholder = st.empty()

    # List of video file paths
    video_files = ["block1.mp4", "block2.mp4", "block3.mp4", "block4.mp4", "block5.mp4"]

    selected_lift = st.sidebar.selectbox("Select Lift", list(range(1, 7)), index=0)

    start_button = st.sidebar.button("Start Counting")

    if start_button:
        for video_file in video_files:
            vid = cv2.VideoCapture(video_file)

            while True:
                ret, frame = vid.read()

                if not ret:
                    break

                # Get count of persons for the selected lift
                count = get_person_count(frame, selected_floor, selected_lift)

                # Display the count in the middle of the screen with styling
                count_placeholder.markdown(
                    f"<div style='font-size: 24px; color: {'green' if count <= 7 else 'red'}; text-align: center;'>Number of Persons in Lift {selected_lift}: {count}</div>",
                    unsafe_allow_html=True,
                )

                # Load person icon based on count and resize
                icon = Image.open(f"{'green' if count <= 7 else 'red'}_person_icon.png")
                icon = icon.resize((50, 50))  # Adjust the size as per your requirement

                # Display the icons
                icon_placeholder.image(icon, width=70)

                # Add a delay between frames
                time.sleep(1)


# Run the main function
if __name__ == "__main__":
    main()
