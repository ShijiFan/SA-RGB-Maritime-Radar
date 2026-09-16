"""Unified, fully auditable Experiment 2 comparison on IPIX1998 across 3 seeds (0, 1, 42).

All models evaluated under the EXACT SAME validation false-alarm calibration:
- 98D Logistic Regression (StandardScaler + LogisticRegression)
- 7D Shared Linear Classifier (shared 7D weights across all 14 cells)
- 7D-32 Shared MLP Classifier (shared 7D-to-32-to-1 scoring across all 14 cells)
- SA-RGB Detector (RadarGrid-Lite / YOLOv11n from formal scratch checkpoints)

Protocol:
1. Training set: 2,916 position-balanced target-present frames.
2. Threshold calibration set: 612 spatially disjoint target-free validation frames.
   Threshold tau is selected to enforce Validation False Alarm Rate <= Cap.
3. Test sets:
   - 684 target-present test frames -> Test Pd (detection requires score >= tau AND correct target cell).
   - 684 spatially disjoint target-free test frames -> Test False Alarm Rate (fraction of empty frames with score >= tau).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = ("Hurst", "RAA", "RDPH", "RVE", "RI", "NR", "MS")


def extract_cell_features(image_path: Path, n_cells: int = 14, cell_pixels: int = 100) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    rows, cols = 2, 7
    half = cell_pixels // 2
    out = np.zeros((n_cells, len(FEATURE_NAMES)), dtype=np.float32)
    for idx in range(n_cells):
        row, col = idx % rows, idx // rows
        y0 = row * cell_pixels
        x0 = col * cell_pixels
        block = image[y0 : y0 + cell_pixels, x0 : x0 + cell_pixels, :]
        top_left = block[:half, :half, :].mean(axis=(0, 1))
        top_right = block[:half, half:, :].mean(axis=(0, 1))
        bottom_left = block[half:, :half, :].mean(axis=(0, 1))
        out[idx] = [
            top_right[0],   # Hurst
            top_left[0],    # RAA
            top_left[1],    # RDPH
            top_left[2],    # RVE
            top_right[1],   # RI
            top_right[2],   # NR
            bottom_left[0], # MS
        ]
    return out / 255.0


def label_to_cell(label_path: Path, rows: int = 2, cols: int = 7) -> int:
    parts = label_path.read_text(encoding="utf-8").strip().split()
    _, x, y, w, h = map(float, parts)
    col = int(round(x * cols - 0.5))
    row = int(round(y * rows - 0.5))
    col = max(0, min(cols - 1, col))
    row = max(0, min(rows - 1, row))
    return col * rows + row


class SharedLinear(nn.Module):
    def __init__(self, in_features: int = 7):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class SharedMLP(nn.Module):
    def __init__(self, in_features: int = 7, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_torch_model(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 150, lr: float = 0.01, seed: int = 42):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(xt, yt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def main():
    root = Path(r"E:\multipletrifeatures")
    ds_pos = root / "paper3_runtime" / "datasets_ipix1998" / "slotrgb_g7_0p512_window14_position_balanced"
    ds_free = root / "paper3_runtime" / "datasets_ipix1998" / "slotrgb_g7_0p512_targetfree14_fullref_valtest"
    rev3 = root / "shengao" / "paper3_jstars_rev3" / "rev3"
    
    # 1. Load data
    print("Loading datasets...")
    x_tr_98, x_tr_cell, y_tr = [], [], []
    for p in sorted((ds_pos / "images" / "train").glob("*.png")):
        c_feats = extract_cell_features(p)
        x_tr_cell.append(c_feats)
        x_tr_98.append(c_feats.reshape(-1))
        y_tr.append(label_to_cell(ds_pos / "labels" / "train" / f"{p.stem}.txt"))
    x_tr_98, x_tr_cell, y_tr = np.asarray(x_tr_98), np.asarray(x_tr_cell), np.asarray(y_tr)
    
    # Val target-free (612 frames)
    x_val_free_98, x_val_free_cell = [], []
    for p in sorted((ds_free / "images" / "val").glob("*.png")):
        c_feats = extract_cell_features(p)
        x_val_free_cell.append(c_feats)
        x_val_free_98.append(c_feats.reshape(-1))
    x_val_free_98, x_val_free_cell = np.asarray(x_val_free_98), np.asarray(x_val_free_cell)
    
    # Test target-present (684 frames)
    x_te_pos_98, x_te_pos_cell, y_test = [], [], []
    for p in sorted((ds_pos / "images" / "test").glob("*.png")):
        c_feats = extract_cell_features(p)
        x_te_pos_cell.append(c_feats)
        x_te_pos_98.append(c_feats.reshape(-1))
        y_test.append(label_to_cell(ds_pos / "labels" / "test" / f"{p.stem}.txt"))
    x_te_pos_98, x_te_pos_cell, y_test = np.asarray(x_te_pos_98), np.asarray(x_te_pos_cell), np.asarray(y_test)
    
    # Test target-free (684 frames)
    x_te_free_98, x_te_free_cell = [], []
    for p in sorted((ds_free / "images" / "test").glob("*.png")):
        c_feats = extract_cell_features(p)
        x_te_free_cell.append(c_feats)
        x_te_free_98.append(c_feats.reshape(-1))
    x_te_free_98, x_te_free_cell = np.asarray(x_te_free_98), np.asarray(x_te_free_cell)
    
    print(f"Data ready: train={len(y_tr)}, val_free={len(x_val_free_98)}, test_pos={len(y_test)}, test_free={len(x_te_free_98)}")
    
    # 2. Read SA-RGB Detector E5 results from file
    df_e5 = pd.read_csv(rev3 / "e5_valcal" / "E5_validation_calibrated_runs.csv")
    caps = [0.10, 0.05, 0.02, 0.01, 0.005]
    seeds = [0, 1, 42]
    
    all_runs = []
    
    for seed in seeds:
        print(f"\n--- Training Seed {seed} ---")
        # 98D Logistic Regression
        pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        pipe_lr.fit(x_tr_98, y_tr)
        lr_classes = pipe_lr.classes_
        
        # Shared Linear
        model_lin = SharedLinear(7)
        train_torch_model(model_lin, x_tr_cell, y_tr, epochs=150, lr=0.02, seed=seed)
        
        # Shared MLP
        model_mlp = SharedMLP(7, 32)
        train_torch_model(model_mlp, x_tr_cell, y_tr, epochs=150, lr=0.01, seed=seed)
        
        def score_lr(x):
            probs = pipe_lr.predict_proba(x)
            return np.max(probs, axis=1), lr_classes[np.argmax(probs, axis=1)]

        def score_lin(x_cell):
            with torch.no_grad():
                logits = model_lin(torch.tensor(x_cell, dtype=torch.float32))
                probs = torch.softmax(logits, dim=-1).numpy()
            return np.max(probs, axis=1), np.argmax(probs, axis=1)

        def score_mlp(x_cell):
            with torch.no_grad():
                logits = model_mlp(torch.tensor(x_cell, dtype=torch.float32))
                probs = torch.softmax(logits, dim=-1).numpy()
            return np.max(probs, axis=1), np.argmax(probs, axis=1)

        models = [
            ("98D Logistic Regression", score_lr, x_val_free_98, x_te_pos_98, x_te_free_98),
            ("Shared Linear Classifier (7D)", score_lin, x_val_free_cell, x_te_pos_cell, x_te_free_cell),
            ("Shared MLP Classifier (7D-32)", score_mlp, x_val_free_cell, x_te_pos_cell, x_te_free_cell),
        ]
        
        for cap in caps:
            for name, fn, val_free_data, te_pos_data, te_free_data in models:
                s_val_free, _ = fn(val_free_data)
                s_te_pos, pred_pos = fn(te_pos_data)
                s_te_free, _ = fn(te_free_data)
                
                tau = float(np.percentile(s_val_free, (1.0 - cap) * 100.0))
                val_fa = float(np.mean(s_val_free >= tau)) * 100.0
                test_pd = float(np.mean((s_te_pos >= tau) & (pred_pos == y_test))) * 100.0
                test_fa = float(np.mean(s_te_free >= tau)) * 100.0
                
                all_runs.append({
                    "Validation_FA_Cap": cap,
                    "Model": name,
                    "Seed": seed,
                    "Selected_Threshold": tau,
                    "Val_FA_Rate(%)": val_fa,
                    "Test_Pd(%)": test_pd,
                    "Test_FA_Rate(%)": test_fa,
                })

    # Add SA-RGB Detector runs from E5
    for cap in caps:
        e5_cap_rows = df_e5[df_e5["validation_false_detection_cap_per_frame"] == cap]
        for _, r in e5_cap_rows.iterrows():
            all_runs.append({
                "Validation_FA_Cap": cap,
                "Model": "SA-RGB Detector",
                "Seed": int(r["seed"]),
                "Selected_Threshold": float(r["selected_threshold"]),
                "Val_FA_Rate(%)": float(r["validation_false_detections_per_frame"]) * 100.0,
                "Test_Pd(%)": float(r["test_Pd"]) * 100.0,
                "Test_FA_Rate(%)": float(r["test_alarm_frame_probability"]) * 100.0,
            })

    # 3. Export detailed and summary tables
    df_all = pd.DataFrame(all_runs)
    df_all.to_csv(rev3 / "unified_experiment2_all_runs.csv", index=False)
    
    # Compute Mean and Std across seeds
    summary_list = []
    for cap in caps:
        sub_cap = df_all[df_all["Validation_FA_Cap"] == cap]
        for model_name in ["98D Logistic Regression", "Shared Linear Classifier (7D)", "Shared MLP Classifier (7D-32)", "SA-RGB Detector"]:
            sub_m = sub_cap[sub_cap["Model"] == model_name]
            pds = sub_m["Test_Pd(%)"].values
            fas = sub_m["Test_FA_Rate(%)"].values
            taus = sub_m["Selected_Threshold"].values
            summary_list.append({
                "Validation_FA_Cap": f"{cap*100:.1f}%",
                "Model": model_name,
                "Test_Pd_Mean(%)": f"{np.mean(pds):.2f}",
                "Test_Pd_Std(%)": f"{np.std(pds):.2f}",
                "Test_FA_Mean(%)": f"{np.mean(fas):.2f}",
                "Test_FA_Std(%)": f"{np.std(fas):.2f}",
                "Mean_Threshold": f"{np.mean(taus):.4f}",
            })
            
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(rev3 / "unified_experiment2_summary.csv", index=False)
    print(f"\nSaved all runs to {rev3 / 'unified_experiment2_all_runs.csv'}")
    print(f"Saved summary to {rev3 / 'unified_experiment2_summary.csv'}")
    print("\n" + "="*90)
    print(df_summary.to_string(index=False))
    print("="*90)


if __name__ == "__main__":
    main()
