import cv2
import numpy as np
import onnxruntime as ort
import torch

# print(torch.cuda.is_available())  # Should return True if GPU is ready
# print(torch.cuda.get_device_name(0))  # Should print GPU name

# Load the ONNX model
onnx_model = ort.InferenceSession("turret_proj/best_human_jetson.onnx", providers=['CUDAExecutionProvider'])

# Open the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

def process_output(output, conf_threshold=0.4, iou_threshold=0.5):
    predictions = np.squeeze(output[0])  # Remove batch dimension (1, 5, 8400) -> (5, 8400)
    predictions = predictions.transpose()  # (5, 8400) -> (8400, 5)
    
    # Get boxes and scores
    boxes = predictions[:, :4]  # First 4 values are bbox coordinates
    scores = predictions[:, 4]  # Fifth value is confidence score
    
    # Filter by confidence
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    if len(boxes) == 0:
        return []
    
    # YOLOv8 outputs boxes in xywh format, scaled to input size (640x640)
    x = boxes[:, 0]  # center x
    y = boxes[:, 1]  # center y
    w = boxes[:, 2]  # width
    h = boxes[:, 3]  # height
    
    # Convert from xywh to xyxy format
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = x - w/2  # x1
    boxes_xyxy[:, 1] = y - h/2  # y1
    boxes_xyxy[:, 2] = x + w/2  # x2
    boxes_xyxy[:, 3] = y + h/2  # y2
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        scores.tolist(),
        conf_threshold,
        iou_threshold
    )
    
    results = []
    if len(indices) > 0:
        indices = indices.flatten()
        for idx in indices:
            box = boxes_xyxy[idx]
            score = scores[idx]
            results.append((box, score))
    
    return results

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Store original dimensions
    original_h, original_w = frame.shape[:2]
    
    # Preprocess the frame
    img = cv2.resize(frame, (640, 640))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name
    outputs = onnx_model.run([output_name], {input_name: img})

    # Process detections
    detections = process_output(outputs)

    # Draw detections
    for box, score in detections:
        # Scale coordinates from model space (640x640) to original frame size
        x1 = int(box[0] * original_w / 640)
        y1 = int(box[1] * original_h / 640)
        x2 = int(box[2] * original_w / 640)
        y2 = int(box[3] * original_h / 640)

        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, original_w - 1))
        y1 = max(0, min(y1, original_h - 1))
        x2 = max(0, min(x2, original_w - 1))
        y2 = max(0, min(y2, original_h - 1))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add confidence score
        label = f"Human: {score:.2f}"
        label_y = max(y1 - 10, 20)  # Ensure label is visible
        cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
