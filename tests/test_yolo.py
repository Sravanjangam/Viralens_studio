from ultralytics import YOLO

model = YOLO("yolov8n.pt")
result = model("https://ultralytics.com/images/bus.jpg")[0]

print("YOLO working!")
print("Classes:", result.boxes.cls.tolist())
print("Conf:", result.boxes.conf.tolist())

