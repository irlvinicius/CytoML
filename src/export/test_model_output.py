import onnxruntime as ort

sess = ort.InferenceSession("backend/models/best.onnx", providers=["CPUExecutionProvider"])         # alterar para path correto
print(sess.get_outputs()[0].shape)

# Esperado: [1, 300, 6]  ← end-to-end YOLOv26
# Se vier: [1, X, 8400]  ← YOLOv11 
