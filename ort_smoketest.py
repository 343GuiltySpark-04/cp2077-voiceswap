# ort_smoketest.py
import os
import time

os.environ.setdefault("ORT_DISABLE_TENSORRT", "1")

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TextColumn,
)
from rich.traceback import install as rich_traceback_install

# Initialize Rich console and better tracebacks
console = Console()
rich_traceback_install()

console.rule("[bold cyan]ONNX Runtime Smoke Test")

# Show ORT version and providers
runtime_table = Table(title="ONNX Runtime Info", show_lines=False)
runtime_table.add_column("Key", style="bold", no_wrap=True)
runtime_table.add_column("Value")
runtime_table.add_row("ORT version", ort.__version__)
runtime_table.add_row("Available providers", ", ".join(ort.get_available_providers()))
console.print(runtime_table)

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

# Session creation with spinner
with console.status("Creating ORT session...", spinner="dots"):
    sess = ort.InferenceSession(model.SerializeToString(), providers=providers)

session_table = Table(title="Session", show_lines=False)
session_table.add_column("Key", style="bold", no_wrap=True)
session_table.add_column("Value")
session_table.add_row("Session providers", ", ".join(sess.get_providers()))
console.print(session_table)

# Prepare inputs
rng = np.random.default_rng(0)
a = rng.standard_normal((256, 256), dtype=np.float32)
b = rng.standard_normal((256, 256), dtype=np.float32)

# Progress for warm-up and benchmarking
progress = Progress(
    SpinnerColumn(spinner_name="dots"),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=None),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
)

runs = 10
latencies: list[float] = []

with progress:
    warm_task = progress.add_task("Warming up", total=1)
    _ = sess.run(None, {"A": a, "B": b})[0]
    progress.advance(warm_task)

    bench_task = progress.add_task(f"Benchmarking ({runs} runs)", total=runs)
    for _ in range(runs):
        t0 = time.perf_counter()
        y = sess.run(None, {"A": a, "B": b})[0]
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        progress.advance(bench_task)

# Metrics
checksum = float(y.sum())
latencies_sorted = sorted(latencies)
avg = sum(latencies_sorted) / len(latencies_sorted)
p50 = latencies_sorted[len(latencies_sorted) // 2]
best = latencies_sorted[0]
worst = latencies_sorted[-1]

# Rough FLOPs for MatMul (m=n=k=256): ~2*m*n*k
ops = 2 * 256 * 256 * 256
gflops = (ops / 1e9) / avg

summary = Table(title="Results", show_lines=False)
summary.add_column("Metric", style="bold")
summary.add_column("Value")
summary.add_row("Output checksum", f"{checksum:.6f}")
summary.add_row("Avg latency (s)", f"{avg:.6f}")
summary.add_row("P50 latency (s)", f"{p50:.6f}")
summary.add_row("Best latency (s)", f"{best:.6f}")
summary.add_row("Worst latency (s)", f"{worst:.6f}")
summary.add_row("Throughput (GFLOP/s, approx)", f"{gflops:.2f}")

console.print(summary)
console.rule("[green]Done")
