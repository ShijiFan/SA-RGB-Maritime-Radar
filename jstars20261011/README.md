# RadarGrid-CS: record-disjoint short-observation detection

Companion to **Short-Observation Seven-Feature Detection of Small Floating Targets in Sea Clutter Under Record-Disjoint Evaluation**, by Shiji Fan, Daniel Albuquerque, Feng Luo, Pedro Pinho, and Ming Shen.

This snapshot accompanies the manuscript prepared for JSTARS submission. It is not a notice of acceptance or publication. The older RadarGrid-Lite implementation elsewhere in this repository uses a different model and evaluation protocol.

## Reproduce stored results

Clone this repository, then run from this directory:

```bash
python -m pip install -r requirements-analysis.txt
python verify_integrity.py
python verify_scores.py
```

The score check recomputes first-design results (manuscript Table VI) from 180 saved model runs, evaluating both tail and fixed thresholds. It also checks 252 final-detector operating-point rows (Table III), from 132 model runs, against logged metrics. A mismatch causes a nonzero exit. The first-design fixed-threshold log uses all 1024 frames; the paper compares thresholds on 1020 common valid frames.

Verified on 2026-10-11: first-design maximum difference 0; final-detector maximum difference 1.11e-16. This is aggregation verification, not new training.

For checkpoint-to-score verification on CPU:

```bash
python -m pip install -r requirements-inference.txt
python example_infer.py
```

Choose a suitable PyTorch build at https://pytorch.org/get-started/locally/ if needed. The example loads G7, K=4, held-out fold 5, seed 0 and predicts a (4, 241, 14) array from the descriptor cache. It compares against stored scores at absolute tolerance 1e-4. The original check returned a maximum difference of 3.81e-6. Neither command starts training.

## Contents

| Directory | Content |
|---|---|
| `data/` | Author-computed descriptor caches and record manifests; no raw radar recordings. |
| `results/` | Per-run metrics and score arrays; `final_detector/` includes equal-dwell controls. |
| `checkpoints/` | First-design and final-model PyTorch state dictionaries. |
| `code/` | Original extraction, experiment, analysis and plotting scripts. |
| `protocol/` | Recorded design decisions and stage-specific results. |
| `Supplementary_Material.pdf` | Figures S1-S3 and Tables S1-S13. |
| `SHA256SUMS.json` | File sizes and SHA-256 digests. |

NPZ feature order: **Hurst, RAA, RDPH, RVE, RI, NR, MCS**. Grid slot order differs; see `code/detector/rgcs.py`. The historical name `MS` means MCS.

## Environment and reproduction scope

Recorded training environment: Python 3.10.19, PyTorch 2.11.0/CUDA 12.8, NumPy 2.2.6, pandas 2.3.3; RTX 5080. Recorded analysis environment: Python 3.14.2, NumPy 2.4.1, SciPy 1.17.0, scikit-learn 1.8.0, pandas 3.0.0.

The two commands above use relative paths. Original experiment scripts retain their development layout and may need path changes. See `REPRODUCTION.txt`. Additional duration caches and source recordings are required for full retraining. A passed score check is not a complete raw-data-to-training reproduction.

Model fitting separates training and test records. Final-design choices used outcomes on these datasets, as documented in the manuscript and `protocol/`. This release is not an additional independent validation set.

## Data and rights

See [data/README.md](data/README.md) and [DATA_NOTICE.txt](DATA_NOTICE.txt). The root MIT license covers the authors' software; it does not license the underlying McMaster recordings.

Contact: Shiji Fan, shiji.fan@ua.pt.
