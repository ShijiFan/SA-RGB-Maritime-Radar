"""Benchmark inference latency of SA-RGB Lite vs Direct G7 Lite on RTX 5080."""

import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, r"e:\multipletrifeatures\first-experments\ultralytics-main\ultralytics-main")
from ultralytics.nn.tasks import DetectionModel


def benchmark_model(model, dummy_input, n_warmup=50, n_iter=500):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        
        # Timed runs
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms
            
    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "fps": float(1000.0 / np.mean(times)),
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # 1. SA-RGB Compact (RadarGrid-Lite)
    cfg_sargb = r"e:\multipletrifeatures\shengao\paper3_jstars_rev3\source\models\paper3_radargrid_lite_safr_singlehead.yaml"
    m_sargb = DetectionModel(cfg=cfg_sargb, ch=3, nc=1, verbose=False).to(device)
    x_sargb = torch.randn(1, 3, 200, 700, device=device)
    
    # 2. Direct G7 Compact
    cfg_d7 = r"e:\multipletrifeatures\paper3_runtime\models\paper3_radargrid_lite_direct7_singlehead.yaml"
    m_d7 = DetectionModel(cfg=cfg_d7, ch=7, nc=1, verbose=False).to(device)
    x_d7 = torch.randn(1, 7, 200, 700, device=device)
    
    print("Benchmarking SA-RGB Lite...")
    bench_sargb = benchmark_model(m_sargb, x_sargb)
    print("SA-RGB Lite:", bench_sargb)
    
    print("Benchmarking Direct G7 Lite...")
    bench_d7 = benchmark_model(m_d7, x_d7)
    print("Direct G7 Lite:", bench_d7)
    
    import json
    res = {
        "device": torch.cuda.get_device_name(0),
        "precision": "FP32",
        "batch_size": 1,
        "input_resolution": "200x700",
        "iterations": 500,
        "SA-RGB_Lite": {
            "channels": 3,
            "params": sum(p.numel() for p in m_sargb.parameters()),
            "stem_params": sum(p.numel() for p in m_sargb.model[0].parameters()),
            "downstream_params": sum(p.numel() for p in m_sargb.parameters()) - sum(p.numel() for p in m_sargb.model[0].parameters()),
            **bench_sargb
        },
        "Direct_G7_Lite": {
            "channels": 7,
            "params": sum(p.numel() for p in m_d7.parameters()),
            "stem_params": sum(p.numel() for p in m_d7.model[0].parameters()),
            "downstream_params": sum(p.numel() for p in m_d7.parameters()) - sum(p.numel() for p in m_d7.model[0].parameters()),
            **bench_d7
        }
    }
    
    out_p = Path(r"e:\multipletrifeatures\shengao\paper3_jstars_rev3\rev3\experiment1_compact_interface_latency.json")
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Saved latency benchmark to {out_p}")


if __name__ == "__main__":
    main()
