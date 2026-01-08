import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def load(self, image_source):
        """
        Runs YOLO on the image.
        image_source = path, URL, or numpy array
        """
        result = self.model(image_source)[0]

        boxes = result.boxes.xyxy.cpu().numpy().tolist()      # [[x1,y1,x2,y2], ...]
        classes = result.boxes.cls.cpu().numpy().tolist()     # [class_ids]
        confs   = result.boxes.conf.cpu().numpy().tolist()    # [confidences]

        # Image size from YOLO
        h, w = result.orig_shape

        # Determine main object → largest bounding box
        main_box = None
        if len(boxes) > 0:
            areas = [
                (b[2] - b[0]) * (b[3] - b[1])
                for b in boxes
            ]
            main_box = boxes[int(np.argmax(areas))]

        return {
            "boxes": boxes,
            "classes": classes,
            "confidences": confs,
            "main_box": main_box,
            "image_size": (w, h),
        }
