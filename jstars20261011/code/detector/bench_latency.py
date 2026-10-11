"""Latency of the frozen G7 cell-scoring network (batch 1, FP32) and of one tail-CFAR threshold update."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from rgcs import RadarGridCS
from run_loro import build_model

HERE = Path(__file__).resolve().parent
res = {}
for dev in ("cuda", "cpu"):
    m = build_model("G7").to(dev).eval()
    x = torch.rand(1, 3, 32, 112, device=dev)
    with torch.no_grad():
        for _ in range(100):
            m(x)
        t = []
        for _ in range(1000):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter(); m(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            t.append((time.perf_counter() - t0) * 1e3)
    res[f"net_{dev}_ms"] = dict(median=float(np.median(t)), p95=float(np.percentile(t, 95)))
# tail-CFAR update for one sequence: reference = previous 64 frames x 13 non-argmax cells
rng = np.random.default_rng(0); R = rng.normal(size=64 * 13)
t = []
for _ in range(2000):
    t0 = time.perf_counter()
    u = np.quantile(R, 0.9); b = (R[R > u] - u).mean(); tau = u + b * np.log(0.1 / (1.0 * 1e-2))
    t.append((time.perf_counter() - t0) * 1e3)
res["tail_cfar_update_ms"] = dict(median=float(np.median(t)), p95=float(np.percentile(t, 95)))
res["params"] = sum(p.numel() for p in build_model("G7").parameters())
res["device"] = torch.cuda.get_device_name(0)
print(json.dumps(res, indent=1))
(HERE / "C_latency.json").write_text(json.dumps(res, indent=1))
