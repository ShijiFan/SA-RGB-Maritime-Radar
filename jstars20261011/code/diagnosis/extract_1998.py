"""Build a 7-feature cache for IPIX1998 (0.512 s, 512 pulses), two reference conventions.

full:   features computed on the whole 28/34-cell record (reference = all other record cells), shape (frames, cells, 7)
window: for every 14-cell window that contains the target (TAES sliding-window rule), features recomputed with the
        reference restricted to that window, shape (frames, n_windows, 14, 7); window starts stored alongside.
Uses the paper3 extractor unchanged (tools/paper3_slotrgb_dataset.extract_features).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, r"E:\multipletrifeatures\tools")
from paper3_slotrgb_dataset import FEATURE_ORDER, META_1998, extract_features, load_complex_pol, ordered_files  # noqa: E402

from common import OUT, POLS

SRC = Path(r"E:\数据\IPIX\1998")
KW = dict(window=512, sample_rate=1000, tf_nperseg=64, tf_noverlap=48, tf_nfft=128, tf_threshold=2.0,
          required_names=FEATURE_ORDER)


def main():
    arrays, records = {}, []
    for rec_no, (path, meta) in enumerate(ordered_files(SRC, META_1998), start=1):
        mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        t = meta.target_cell - 1
        for pol in POLS:
            data = load_complex_pol(mat, pol)
            n = data.shape[0]
            full = extract_features(data=data, **KW)
            starts = list(range(max(0, t - 13), min(t, n - 14) + 1))
            win = np.stack([extract_features(data=data[s:s + 14], **KW) for s in starts], axis=1)
            key = f"r{rec_no:02d}_{pol}"
            arrays[key + "_full"] = full.astype(np.float32)
            arrays[key + "_win"] = win.astype(np.float32)
            arrays[key + "_starts"] = np.array(starts)
            print(key, path.name, n, full.shape, win.shape, flush=True)
        records.append(dict(rec=rec_no, stem=meta.stem, file=str(path), cells=int(n), target_cell=meta.target_cell))
    np.savez_compressed(OUT / "ipix1998_7feat.npz", **arrays)
    json.dump(dict(feature_order=list(FEATURE_ORDER), params={k: v for k, v in KW.items() if k != "required_names"},
                   records=records), open(OUT / "ipix1998_7feat.json", "w", encoding="utf8"), indent=1)


if __name__ == "__main__":
    main()
