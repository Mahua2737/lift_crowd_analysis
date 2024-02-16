"""
1. pip install setuptools
2. pip install numpy opencv-python 
3. pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
4. Download video from youtube.
5. Refer the code and run.
"""
import torch
import cv2  # Image processing
import datetime

model = torch.hub.load(
    "ultralytics/yolov5", "yolov5s"
)  # Loading the yolov5s model into memory of computer CPU.
# Read about yolov5 and tell what it is not depth required.

vid = cv2.VideoCapture("block1.mp4")  # Loading the video in the memory for reading

while True:  # Looping through the video until it gets over
    ret, frame = vid.read()  # Fetch frame from video one by one
    if ret == False:  # If frame is empty we are breaking the loop.
        break
    results = model(
        frame
    )  # Passing the frame into the model, frame is in the form of opencv matrix.
    detections = []
    if not results.pandas().xyxy[0].empty:
        for result in results.pandas().xyxy[0].itertuples():
            if result[7] == "person":
                detections.append(
                    [int(result[1]), int(result[2]), int(result[3]), int(result[4])]
                )  # Appending the person's coordinate for the boxes.

    # Iterating through the person and drawing boxes over them
    for det in detections:
        c1, c2 = (int(det[0]) - 10, int(det[1])), (int(det[2]) + 10, int(det[3]))
        color = (0, 0, 255)
        cv2.rectangle(frame, c1, c2, color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            "Number of Person: {}".format(len(detections)),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
        )

    print(datetime.datetime.now(), len(detections))
    cv2.imshow("test", frame)  # Show the frame with the boxes
    cv2.waitKey(1)  # Add 1 second delay between frames.

"""
1. What have you learnt while doing it ? 
-- Learnt how to setup opencv and torch dependencies, understand how to read videos/images in computer as a data 
( images are nothing but matrix), what is object detection, what is yolo and how to run it on laptop.

2. What parts were done by you and your teammate ?
-- Since AI is something of a new domain for us, we both are learning and doing these blocks of code together.

3. What kind of system does it require to run?
-- Minimum requirement is something we have not decided but so far minimum 8gb of ram is atleast being used to run this code.

4. How are we going to access it ?
-- We can use this code in a form of web page or any display, for this we have to deploy it on cloud but currently we are running it on laptop
so we can see these output on screen.
"""
