"""TensorRT FP16 throughput of an exported detector (ONNX), for the hardware sizing: builds an engine
with a dynamic batch and times it on CUDA buffers from PyTorch.

    python ml/bench_trt.py --model $RUNS/export/det_large/rfdetr-large.onnx --batch 1 8 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorrt as trt
import torch

DTYPES = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(args.model)):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    inp = network.get_input(0)
    _, c, h, w = inp.shape
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    profile = builder.create_optimization_profile()
    top = max(args.batch)
    profile.set_shape(inp.name, (1, c, h, w), (top, c, h, w), (top, c, h, w))
    config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()
    gpu = torch.cuda.get_device_name()
    results = []
    for b in args.batch:
        ctx.set_input_shape(inp.name, (b, c, h, w))
        buffers = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            buffers[name] = torch.zeros(
                tuple(ctx.get_tensor_shape(name)), dtype=DTYPES[engine.get_tensor_dtype(name)], device="cuda"
            )
            ctx.set_tensor_address(name, buffers[name].data_ptr())
        stream = torch.cuda.Stream()
        for _ in range(10):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        t = time.perf_counter()
        for _ in range(args.iters):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        ms = (time.perf_counter() - t) / args.iters * 1000
        results.append({"batch": b, "ms_per_batch": ms, "frames_per_s": b * 1000 / ms})
        print(f"{gpu} batch {b:3d}: {ms:7.2f} ms, {b * 1000 / ms:7.1f} frames/s (TensorRT FP16)")
    print(json.dumps({"gpu": gpu, "model": str(args.model), "input": [h, w], "build_s": build_s, "results": results}))


if __name__ == "__main__":
    main()
