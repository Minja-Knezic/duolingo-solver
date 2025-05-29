from ultralytics import YOLO
import os


model = YOLO("yolov8n.pt")  

script_dir = os.path.dirname(os.path.abspath(__file__))  
project_path = os.path.join(script_dir, "runs")

model.train(
    data="duolingo.yaml",
    epochs=10,
    imgsz=416,
    batch=8,   
    project=project_path,      
    name="duolingo_yolo"
)
