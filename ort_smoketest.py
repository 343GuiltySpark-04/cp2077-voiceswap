# ort_smoketest.py
import os

os.environ.setdefault("ORT_DISABLE_TENSORRT", "1")

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

print("ORT version:", ort.__version__)
print("ORT available providers:", ort.get_available_providers())

# Build tiny graph: Y = A @ B
A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [256, 256])
B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [256, 256])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [256, 256])

node = helper.make_node("MatMul", ["A", "B"], ["Y"])
graph = helper.make_graph([node], "mm", [A, B], [Y])

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

# Pin IR version to something your ORT accepts
model.ir_version = 11

onnx.checker.check_model(model)

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession(model.SerializeToString(), providers=providers)

print("ORT session providers:", sess.get_providers())

a = np.random.randn(256, 256).astype(np.float32)
b = np.random.randn(256, 256).astype(np.float32)
y = sess.run(None, {"A": a, "B": b})[0]
print("Output checksum:", float(y.sum()))
