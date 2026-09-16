"""Train Direct G7 Lite (paper3_radargrid_lite_direct7_singlehead) with strictly controlled hyperparameters.

Hyperparameters strictly matched with SA-RGB Lite (formal_lite_g7_e40_b8_multiseed):
- Architecture: paper3_radargrid_lite_direct7_singlehead.yaml (325,033 params)
- Dataset: E:/multipletrifeatures/paper3_runtime/datasets_tensor/tensor_g7_0p512_v2/data.yaml
- Epochs: 40
- Batch: 8
- Imgsz: 640
- Optimizer: AdamW
- Cos_lr: True
- Device: '0'
- Workers: 0
- Cache: 'ram'
"""

import argparse
import sys
import time
from pathlib import Path

# Add ultralytics path
sys.path.insert(0, r"e:\multipletrifeatures\first-experments\ultralytics-main\ultralytics-main")
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    model_cfg = r"E:\multipletrifeatures\paper3_runtime\models\paper3_radargrid_lite_direct7_singlehead.yaml"
    data_yaml = r"E:\multipletrifeatures\paper3_runtime\datasets_tensor\tensor_g7_0p512_v2\data.yaml"
    project_dir = r"E:\multipletrifeatures\paper3_runtime\runs\formal_direct7_lite_g7_e40_b8_multiseed"
    run_name = f"direct7_radargrid_lite_singlehead_e40_b8_s{args.seed}"

    print(f"=== Starting Direct G7 Lite Training: Seed {args.seed}, Epochs {args.epochs}, Batch {args.batch} ===")
    t0 = time.time()
    model = YOLO(model_cfg)
    
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        cache="ram",
        imgsz=640,
        device=args.device,
        workers=0,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        pretrained=False,
        optimizer="AdamW",
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"=== Completed Direct G7 Lite Seed {args.seed} in {elapsed:.1f} s ({elapsed/60:.2f} min) ===")

if __name__ == "__main__":
    main()
