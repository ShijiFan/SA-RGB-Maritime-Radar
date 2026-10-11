"""Single-checkpoint inference example (needs torch, numpy, pandas, scikit-learn, scipy).

Loads the final detector of LORO fold 5, seed 0 (G7, K = 4), recomputes its test-record scores from the IPIX1993
descriptor cache in data/, and compares them with the stored score array.  Run from the unzipped folder:
    python example_infer.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for sub in ("code/diagnosis", "code/detector", "code/temporal"):
    sys.path.insert(0, str(HERE / sub))
import common  # noqa: E402

common.CACHE = HERE / "data"                         # descriptor caches shipped with this archive
import n1_temporal_net as N  # noqa: E402
N.DEV = "cpu"
import n3_final  # noqa: E402
n3_final.DEV = "cpu"
from rgcs import fit_percentiles  # noqa: E402

FOLD, SEED = 5, 0
data = common.load("0p512")
train, val, test = common.loro_split(FOLD)
lo, hi = fit_percentiles(np.concatenate([data[g]["X"].reshape(-1, 7) for g in train]))   # 8-bit range of the fold
feat = n3_final.make_feat("G7")
F = feat(data[test]["X"], lo, hi).cpu()                                                 # (4 pol, 256 frames, 14, 72)
model = N.TemporalCellNet()
model.load_state_dict(torch.load(HERE / "checkpoints" / "final" / f"G7_K4_f{FOLD:02d}_s{SEED}.pt", map_location="cpu", weights_only=True))
model.eval()
scores = N.net_scores(model, F, np.arange(15, 256), 4)                                 # logits, frames 15-255
ref = np.load(HERE / "results" / "final_detector" / "scores" / f"G7_K4_f{FOLD:02d}_s{SEED}_scores.npz")["test"]
if scores.shape != ref.shape or not np.isfinite(scores).all():
    raise SystemExit("inference verification FAILED: shape or finite-value check")
error = float(np.abs(scores - ref).max())
print("score array", scores.shape, "max |difference| to stored scores:", error)
if not np.isfinite(error) or error > 1e-4:
    raise SystemExit("inference verification FAILED: absolute tolerance exceeded")
print("inference verification passed")
