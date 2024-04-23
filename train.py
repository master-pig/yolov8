from ultralytics import YOLO

model = YOLO("yolov8x-pose.pt")

model.train(data="yolo_strawberry_keypoints.yaml",workers=0,epochs=50,batch=16) 