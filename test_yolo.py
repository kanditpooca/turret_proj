from ultralytics import YOLO
import cv2
import torch
import numpy as np

def main():
    # Load your trained model
    model = YOLO("turret_proj/best_human.engine")  # Load the trained weights
    
    # Enable tracking in the model
    model.tracker = "bytetrack.yaml"  # Using ByteTrack algorithm
    
    # Use webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Get the center of the frame for tracking reference
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = frame_width // 2
    center_y = frame_height // 2
    
    while cap.isOpened():
        # Read a frame from the camera
        success, frame = cap.read()
        
        if success:
            # Run YOLOv8 inference with tracking
            results = model.track(frame, conf=0.4, persist=True)
            
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()  # Get boxes in xywh format
                track_ids = results[0].boxes.id.cpu()  # Get track IDs
                
                # Draw tracking info for each detection
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    
                    # Calculate box center
                    box_center_x = int(x)
                    box_center_y = int(y)
                    
                    # Calculate offset from frame center
                    offset_x = box_center_x - center_x
                    offset_y = box_center_y - center_y
                    
                    # Draw tracking info
                    cv2.circle(frame, (box_center_x, box_center_y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"ID: {int(track_id)}", (box_center_x, box_center_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Offset X: {int(offset_x)} Y: {int(offset_y)}", 
                              (box_center_x, box_center_y + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Draw center crosshair
            cv2.line(annotated_frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
            cv2.line(annotated_frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
            
            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
