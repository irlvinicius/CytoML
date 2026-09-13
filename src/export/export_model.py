from ultralytics import YOLO

model = YOLO("path_do_modelo_pt")
model.export(format="onnx", imgsz=640, simplify=True)           # gera models/best.onnx
