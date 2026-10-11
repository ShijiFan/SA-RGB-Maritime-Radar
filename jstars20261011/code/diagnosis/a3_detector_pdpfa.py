"""A3: radar operating points of the historical SA-RGB G3/G7 LRO detectors (YOLO11n, seed 42).

Infers every fold checkpoint on its validation-record and test-record images, maps boxes to cell scores
(max confidence of boxes centred in the cell, conf floor 1e-3, NMS IoU 0.7 -- same as R1 and TAES), calibrates
tau on the validation record's H0 cells and applies it to the test record.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "rev3" / ".ultralytics"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_r1_lro_common_metric import center_to_cell, target_from_label  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from common import OUT, calibrate, pd_at_pfa  # noqa: E402

ALPHAS = (1e-2, 1e-3)
NAME = re.compile(r"(hh|hv|vh|vv)_group(\d+)_.*_(\d+)\.png$", re.I)


def infer(model, root, split):
    imgs = sorted((root / "images" / split).glob("*.png"))
    S = np.zeros((len(imgs), 14))
    t = np.array([target_from_label(root / "labels" / split / f"{p.stem}.txt") for p in imgs])
    meta = [NAME.search(p.name).groups() for p in imgs]
    for i, r in enumerate(model.predict(source=[str(p) for p in imgs], imgsz=640, batch=32, device=0, conf=1e-3,
                                        iou=0.7, max_det=300, stream=True, verbose=False)):
        if r.boxes is not None and len(r.boxes):
            for c, s in zip(r.boxes.xywhn[:, :2].cpu().numpy(), r.boxes.conf.cpu().numpy()):
                k = center_to_cell(float(c[0]), float(c[1]))
                S[i, k] = max(S[i, k], float(s))
    pol = np.array([m[0].upper() for m in meta]); rec = np.array([int(m[1]) for m in meta])
    frame = np.array([int(m[2]) for m in meta])
    return S, t, pol, rec, frame


def main():
    runs = pd.read_csv(ROOT / "rev3" / "R1_lro_common_metric.csv")
    runs = runs[runs.method.str.startswith("SA-RGB")]
    rows, store = [], {}
    for _, r in runs.iterrows():
        root = Path(r.dataset_yaml).parent
        model = YOLO(r.checkpoint)
        Sv, tv, *_ = infer(model, root, "val")
        St, tt, pol, rec, frame = infer(model, root, "test")
        key = f"{r.method.split()[-1]}_f{int(r.fold):02d}"
        store.update({key + "_val": Sv, key + "_valt": tv, key + "_test": St, key + "_testt": tt,
                      key + "_pol": pol, key + "_frame": frame})
        h0v = Sv[np.arange(14)[None] != tv[:, None]]
        h1t = St[np.arange(len(tt)), tt]; h0t = St[np.arange(14)[None] != tt[:, None]]
        adj = St[np.abs(np.arange(14)[None] - tt[:, None]) == 1]  # index neighbours in the 2x7 cell numbering
        row = dict(method=r.method, fold=int(r.fold), exact_cell=float(np.mean(St.argmax(1) == tt)),
                   val_frac_h0_boxed=float(np.mean(h0v > 0)), test_frac_h0_boxed=float(np.mean(h0t > 0)))
        for a in ALPHAS:
            tau = calibrate(h0v, a)
            row.update({f"tau_{a:g}": tau, f"pfa_{a:g}": float(np.mean(h0t >= tau)),
                        f"pfa_adj_{a:g}": float(np.mean(adj >= tau)), f"pd_{a:g}": float(np.mean(h1t >= tau)),
                        f"pdor_{a:g}": pd_at_pfa(h1t, h0t, a)})
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    np.savez_compressed(OUT / "A3_detector_cell_scores.npz", **store)
    pd.DataFrame(rows).to_csv(OUT / "A3_detector_pdpfa.csv", index=False)


if __name__ == "__main__":
    main()
