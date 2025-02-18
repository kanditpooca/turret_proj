from typing import Tuple, Union
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    # for keypoint in detection.keypoints:
    #   keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
    #                                                  width, height)
    #   color, thickness, radius = (0, 255, 0), 2, 2
    #   cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw navigation tracking line
    x_centerImg = width // 2
    y_centerImg = height // 2

    # Find center point of the bounding box
    x_startBox, y_startBox = start_point
    x_endBox, y_endBox = end_point
    x_centerBox = (x_startBox + x_endBox)//2
    y_centerBox = (y_startBox + y_endBox)//2

    # Draw tracking navigation line
    cv2.arrowedLine(annotated_image, (x_centerImg, y_centerImg), (x_centerBox,y_centerBox), (0,255,0),2)

    # Calculate error
    X_error = x_centerBox -  x_centerImg
    Y_error = -(y_centerBox - y_centerImg)

    cv2.putText(annotated_image, f"X_error: {X_error} px", (width - 300, height - 100), FONT, 0.8, [0,255,0], 2)
    cv2.putText(annotated_image, f"Y_error: {Y_error} px", (width - 300, height - 50), FONT, 0.8, [0,255,0], 2)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

def show_referenceAxis(image):
  # X-axis
  cv2.line(image, (0, height//2), (width, height//2), (0,255,0),1)

  # Y-axis
  cv2.line(image, (width//2, 0), (width//2, height), (0,255,0),1)

# Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# Get the frames from camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Get width and height of the frame
    height, width, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    detection_result = detector.detect(mp_image)

    # Process the detection result. In this case, visualize it.
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    show_referenceAxis(rgb_annotated_image)

    cv2.imshow("Face detection", rgb_annotated_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()