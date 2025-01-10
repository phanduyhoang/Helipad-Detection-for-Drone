import cv2
import os
from ultralytics import YOLO
import supervision as sv

# ---------------------- Configuration ----------------------
YOLO_MODEL_PATH = r"C:\Users\admin\Downloads\car_detection\HELIPAD_DETECTION\best.pt"
CONFIDENCE_THRESHOLD = 0.1
VIDEO_SOURCE = r'C:\Users\admin\Downloads\car_detection\HELIPAD_DETECTION\video_drone.mp4'

# ---------------------- Initialize Paths ----------------------
# Get directory of the input video
video_dir = os.path.dirname(VIDEO_SOURCE)
output_video_path = os.path.join(video_dir, "output_video_drone.mp4")

# ---------------------- Initialization ----------------------
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
model = YOLO(YOLO_MODEL_PATH)
print("Loaded YOLO model with classes:", model.names)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Unable to open video source {VIDEO_SOURCE}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# ---------------------- Processing Loop ----------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot fetch the frame.")
        break

    # Convert frame to RGB (YOLO expects RGB images)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO prediction
    results = model.predict(image_rgb, conf=CONFIDENCE_THRESHOLD)
    detections = results[0]
    
    # Convert YOLO detections to Supervision format
    detections = sv.Detections.from_ultralytics(detections)

    # Update tracker with current frame detections
    detections = tracker.update_with_detections(detections)

    # Prepare labels with tracker ID and class name
    labels = [
        f'ID: {tracker_id} | {model.names[class_id]}'
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
    ]

    # Draw bounding boxes
    annotated_frame = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)

    # Draw labels (tracker IDs and class names)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Convert frame back to BGR for OpenCV display and saving
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Write the frame to the output video file
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("YOLO Detection with Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# ---------------------- Cleanup ----------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved at: {output_video_path}")
