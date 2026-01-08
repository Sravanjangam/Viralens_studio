from cv_engine.detector import ObjectDetector

detector = ObjectDetector()

result = detector.load("https://ultralytics.com/images/bus.jpg")

print("Keys:", result.keys())
print("Num Boxes:", len(result["boxes"]))
print("Main Box:", result["main_box"])
print("Image Size:", result["image_size"])
