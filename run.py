import cv2
from ultralytics import YOLO
import supervision as sv

tracker=sv.Bytetrack()
box_annotator=sv.BoundingBoxAnnotator()
label_annotator=sv.LabelAnnotator()

# Load the trained YOLO model
model = YOLO(r"C:\Users\admin\Downloads\car_detection\HELIPAD_DETECTION\best.pt")
 # Make sure "best.pt" is your trained model file
print(model.names)


cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:  
        break

    # Convert frame to RGB (YOLO expects RGB images)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO prediction
    results = model.predict(image_rgb, conf=0.4)

    # Get frame with bounding boxes
    detections = results[0]
    detections=sv.Detections.from_ultralytics(detections)
    detections=tracker.update_with_detections(detections)

    labels=[
        f'# {tracker_id} {results.names[class_id]}'
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame=box_annotator.annotate(
        image_rgb.copy(), detections=detections, 
    )

    annotated_frame=cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    # Display the frame with detections
    cv2.imshow("YOLO Detection", annotated_frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
