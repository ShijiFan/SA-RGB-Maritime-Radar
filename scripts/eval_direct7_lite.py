"""Validation threshold calibration and test evaluation for Direct G7 Lite.
Uses the EXACT same protocol, metric definitions, and CaptureValidator as run_p0_common_operating.py.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"e:\multipletrifeatures\first-experments\ultralytics-main\ultralytics-main")
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import compute_ap

class CaptureValidator(DetectionValidator):
    capture_path = None
    def init_metrics(self, model):
        super().init_metrics(model)
        self.captured = []
        self.roster = []
        
    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            b = self._prepare_batch(si, batch)
            assert len(b['cls']) == 1
            name = Path(b['im_file']).stem
            self.roster.append(name)
            tp = self._process_batch(pred, b)['tp']
            conf = pred['conf'].detach().cpu().numpy()
            metric_boxes = pred['bboxes'].detach().cpu().numpy().copy()
            metric_gt = b['bboxes'].detach().cpu().numpy()[0].copy()
            scaled = self.scale_preds(pred, b)
            xyxy = scaled['bboxes'].detach().cpu().numpy()
            h, w = b['ori_shape']
            for j, c in enumerate(conf):
                row = {'image': name, 'score': float(c)}
                row.update({f'tp{k}': int(tp[j, k]) for k in range(10)})
                row.update(zip(('x1', 'y1', 'x2', 'y2'), map(float, xyxy[j] / np.array([w, h, w, h]))))
                row.update(zip(('metric_x1', 'metric_y1', 'metric_x2', 'metric_y2'), map(float, metric_boxes[j])))
                row.update(zip(('gt_metric_x1', 'gt_metric_y1', 'gt_metric_x2', 'gt_metric_y2'), map(float, metric_gt)))
                self.captured.append(row)
        super().update_metrics(preds, batch)
        
    def finalize_metrics(self):
        super().finalize_metrics()
        assert len(set(self.roster)) == len(self.roster)
        self.capture_path.with_suffix('.frames.json').write_text(json.dumps(self.roster))
        with open(self.capture_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.captured[0]))
            w.writeheader()
            w.writerows(self.captured)


def arrays(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    roster = json.loads(path.with_suffix('.frames.json').read_text())
    assert set(r['image'] for r in rows) <= set(roster)
    scores = np.array([float(r['score']) for r in rows])
    tps = np.array([[int(r[f'tp{k}']) for k in range(10)] for r in rows])
    return scores, tps, len(roster)


def threshold(scores, tp, n, cap):
    order = np.argsort(-scores, kind='stable')
    sc = scores[order]
    t = tp[order]
    ends = np.r_[np.where(sc[:-1] != sc[1:])[0], len(sc) - 1]
    cum = t.cumsum()
    fp = np.arange(1, len(sc) + 1) - cum
    feasible = ends[fp[ends] <= cap * n + 1e-10]
    if not len(feasible):
        return float(np.nextafter(sc.max(), np.inf))
    best = sorted(feasible, key=lambda i: (-cum[i], fp[i], -sc[i]))[0]
    if cum[best] == 0:
        return float(np.nextafter(sc.max(), np.inf))
    return float(sc[best])


def evaluate_checkpoint(weight_path: Path, seed: int, out_dir: Path, data_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    val_csv = out_dir / f"direct7_lite_s{seed}_val.csv"
    test_csv = out_dir / f"direct7_lite_s{seed}_test.csv"
    
    # 1. Run Val
    CaptureValidator.capture_path = val_csv
    model = YOLO(str(weight_path))
    model.val(
        validator=CaptureValidator,
        data=str(data_dir / "data.yaml"),
        split="val",
        imgsz=640,
        batch=8,
        device="0",
        workers=0,
        half=False,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=False,
        verbose=False,
        project=str(out_dir / "validator"),
        name=f"val_s{seed}",
        exist_ok=True,
    )
    val_csv.with_suffix('.provenance.json').write_text(json.dumps(dict(
        checkpoint=str(weight_path),
        checkpoint_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        dataset=str(data_dir),
        split="val",
    ), indent=2))

    # 2. Run Test
    CaptureValidator.capture_path = test_csv
    model = YOLO(str(weight_path))
    model.val(
        validator=CaptureValidator,
        data=str(data_dir / "data.yaml"),
        split="test",
        imgsz=640,
        batch=8,
        device="0",
        workers=0,
        half=False,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=False,
        verbose=False,
        project=str(out_dir / "validator"),
        name=f"test_s{seed}",
        exist_ok=True,
    )
    test_csv.with_suffix('.provenance.json').write_text(json.dumps(dict(
        checkpoint=str(weight_path),
        checkpoint_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        dataset=str(data_dir),
        split="test",
    ), indent=2))

    # 3. Summarize operating points
    vs, vt, vn = arrays(val_csv)
    ts, tt, tn = arrays(test_csv)
    assert (vn, tn) == (1800, 1200)
    
    order = np.argsort(-ts, kind='stable')
    aps = []
    for k in range(10):
        cum = tt[order, k].cumsum()
        aps.append(float(compute_ap(cum / tn, cum / np.arange(1, len(ts) + 1))[0]))
    
    summary_rows = []
    for cap in [0.001, 0.01]:
        th = threshold(vs, vt[:, 0], vn, cap)
        vm = vs >= th
        tm = ts >= th
        vtp = int(vt[vm, 0].sum())
        ttp = int(tt[tm, 0].sum())
        summary_rows.append(dict(
            model="direct7_lite",
            seed=seed,
            val_cap=cap,
            threshold=th,
            val_images=vn,
            val_TP=vtp,
            val_FP=int(vm.sum()) - vtp,
            test_images=tn,
            test_TP=ttp,
            test_FN=tn - ttp,
            test_FP=int(tm.sum()) - ttp,
            test_Pd=ttp / tn,
            test_FPPI=(int(tm.sum()) - ttp) / tn,
            test_AP50=aps[0],
            test_AP50_95=float(np.mean(aps)),
        ))
    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    out_dir = Path(r"E:\multipletrifeatures\shengao\paper3_jstars_rev3\rev3\p0_common_operating_20260915\direct7_lite_eval")
    data_dir = Path(r"E:\multipletrifeatures\paper3_runtime\datasets_tensor\tensor_g7_0p512_v2")
    res = evaluate_checkpoint(Path(args.checkpoint), args.seed, out_dir, data_dir)
    print("\nEvaluation Results for Direct G7 Lite:")
    for r in res:
        print(f"Cap {r['val_cap']}: Threshold={r['threshold']:.4f}, Test Pd={r['test_Pd']*100:.2f}%, Test FPPI={r['test_FPPI']:.4f}, AP50={r['test_AP50']*100:.2f}%, AP50-95={r['test_AP50_95']*100:.2f}%")
