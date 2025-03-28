from ultralytics import YOLO
from multiprocessing import freeze_support
import torch

if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8s model try yolov8n.pt first (smallest version)
    
    # Configure training with settings optimized for tracking
    model.train(
        data="turret_proj/datasets/mot17_filtered/data.yaml",  # Using our filtered dataset
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,
        cache=False,
        device=0 if torch.cuda.is_available() else 'cpu',
        close_mosaic=10,
        patience=50,  # Early stopping patience
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,  # Enable augmentation
        degrees=5.0,  # Slight rotation augmentation
        scale=0.5,    # Scale augmentation
        perspective=0.0007,  # Perspective augmentation
        flipud=0.0,   # No vertical flip for tracking
        fliplr=0.5,   # Horizontal flip
        mosaic=0.5,   # Mosaic augmentation
        mixup=0.1     # Mixup augmentation
    )
