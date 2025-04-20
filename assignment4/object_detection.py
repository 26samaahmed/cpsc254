import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load YOLOv5 which is a pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img_path = 'objects.jpg'
img = Image.open(img_path)
results = model(img) # Perform inference

# Extract labels and coordinates from the results
labels = results.names
predicted_labels = results.names
predictions = results.xywh[0]

# Print the detected object names and their locations
print("Predicted Object Labels and Their Coordinates:")
for pred in predictions:
    class_id = int(pred[5])
    class_name = labels[class_id]
    x_center, y_center, width, height = pred[:4]
    print(f"Object: {class_name}, Coordinates: ({x_center:.2f}, {y_center:.2f}, {width:.2f}, {height:.2f})")

img = cv2.imread(img_path)

for pred in predictions:
    x_center, y_center, width, height = pred[:4]
    x1 = int((x_center - width / 2) * img.shape[1])
    y1 = int((y_center - height / 2) * img.shape[0])
    x2 = int((x_center + width / 2) * img.shape[1])
    y2 = int((y_center + height / 2) * img.shape[0])

    # Drawing the bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Showing the image with bounding boxes
cv2.imshow("Object Detection", img)
cv2.waitKey(1000)  # Wait for 1 second
cv2.destroyAllWindows()
