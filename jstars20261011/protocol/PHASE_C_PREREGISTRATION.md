# Phase C pre-registration: SA-RGB cell-scoring network with clutter-map tail CFAR

Status: **FROZEN.** The author approved the protocol in chat on 2026-10-08 ("可以，直接开始做吧"), before any Phase C run that computes test metrics. The only earlier Phase C run was a smoke test (fold 1, seed 0, G7) that printed losses only.
Once signed, this file is frozen. Any later change goes in the amendment log at the end, with a date and a reason, and it must be made before the affected run.

## 1. Questions

- **Q1 (information).** Under record-disjoint evaluation, does the seven-feature SA-RGB input (G7) give a higher detection probability than the three-feature input (G3) at a calibrated false-alarm budget, when both use the same network?
- **Q2 (controllability).** Does G7 keep the realized test false-alarm rate as close to the budget as G3, or closer?
- **Q3 (transfer).** Do Q1 and Q2 hold on IPIX1998 genuine sliding windows, using models trained only on IPIX1993?

## 2. Development history (to be disclosed in the paper)

Phases A and B (`phaseA_20261008/`, `PROGRESS.md`) used IPIX1993 LORO test folds to diagnose failures. They also selected the following on those folds:
- per-frame slot normalization;
- relative context;
- the exponential-tail clutter-map rule.

IPIX1998 was used once, in A6, with position-free tree and linear scorers. It was not used in any module selection. Phase C on IPIX1993 is therefore a **confirmatory replication with a new model class**, not a fully independent test. IPIX1998 (Section 6) is the cleaner test, and it is run exactly once.

## 3. Data and input

- **Features.** The seven paper3 descriptors at 0.512 s. IPIX1993 uses `rev3/cache_e3/raw_features_0p512.npz`. IPIX1998 uses `phaseA_20261008/out/ipix1998_7feat.npz`, window-local reference.
- **SA-RGB rendering.**
  - Fixed paper3 address map: s1 = (RAA, RDPH, RVE), s2 = (Hurst, RI, NR), s3 = (MCS, a, a), s4 = (a, a, a), with a = 128/255. G3 fills s1 only and leaves the other slots at a.
  - Quantization: 8 bit, 1st/99th percentiles fitted on the fold's training records only.
  - Macro-cell size is 16 x 16 px and each slot is 8 x 8 px, giving a 32 x 112 image in the 2 x 7 odd-even layout. Each block is constant, so this resolution loses no information relative to 100 px; this point is stated in the paper.
- **IPIX1998.** The 1993 training percentiles are reused unchanged, as a deployment-style rule with no target-site refit.

## 4. Network (RadarGrid-CS: cell-scoring)

1. **Anchor lift (SAFR).** [I, I - a, |I - a|].
2. **Slot readout.** Average pooling over each 8 x 8 slot, then reshaping to 12 addresses per macro-cell on the 2 x 7 cell grid.
3. **Slot clutter normalization (SCN).** For each address and frame: (x - median over the 14 cells) / (IQR over the 14 cells + eps). Addresses with zero IQR in a frame are set to 0.
4. **Relative context (RC).** For each of the 8 neighbours in the 3 x 3 cell window, the difference x_c - x_nb, a validity mask for neighbours outside the grid (difference set to 0), and the peak margin x_c - max(valid neighbours).
5. **Shared cell head.** 1 x 1 convolutions only: input -> 64 -> 64 -> 1, with SiLU and dropout 0.1. The weights are shared across cells and there is no absolute-position input.
6. **Training.**
   - Loss: per-cell binary cross-entropy with pos_weight 13.
   - Optimizer: AdamW, lr 1e-3, weight decay 1e-4, batch 256 frames, 40 epochs.
   - Checkpoint selection: the minimum validation-record BCE.
   - No augmentation.
7. **Output CFAR (texp64).** Per test sequence (record x polarization; for IPIX1998, record x polarization x window start), a causal reference is built from the previous 64 frames' non-argmax cell logits.
   - Exponential tail above the reference 90th percentile: tau(a) = u + beta * ln(0.1 / (c a)).
   - The scale c is calibrated once on the validation sequences at a = 1e-2 and reused at 1e-3.
   - The first frame of each sequence has no history and is excluded, as are the frames with an infinite threshold.

## 5. Arms (identical training and CFAR unless stated)

| Arm | Input | Purpose |
|---|---|---|
| **G7** | SA-RGB, 7 features | primary |
| **G3** | SA-RGB, 3 features | primary comparator |
| D7 | raw 7-vector per cell, same SCN, RC and head (no image) | shows whether the image adds or loses anything |
| P7 | learned linear 7 -> 3 projection, then the same pipeline | linear-compression comparator |
| G7-noSCN | G7 without SCN | ablation |
| G7-absC | G7 with absolute neighbour values in place of RC | ablation |
| G7-fixed | G7 with a fixed validation threshold in place of texp64 | ablation of the CFAR rule |
| References | historical YOLO11n SA-RGB G3/G7 (A3, one seed); HGB-C2 G3/G7 (Phase B) | context only; not part of any decision |

## 6. Protocols

- **IPIX1993 LORO.**
  - Fold k tests on record k and validates on record k+1 (record 9 for k = 10); the remaining 8 records train.
  - Seeds 0, 1, 2.
  - All four polarizations are pooled. H0 is the 13 non-target cells of each frame; "adjacent" means the two range neighbours.
- **IPIX1998 cross-site (run once, after all IPIX1993 runs are complete and summarized).**
  - Train on the chronological IPIX1993 training frames (the first 181 of 256 per sequence, all 10 records).
  - Select the checkpoint and calibrate c on the IPIX1993 validation frames (the next 45).
  - Test on all 23,868 IPIX1998 windows. Seeds 0, 1, 2. Arms: G7, G3 and D7 only.

## 7. Endpoints and decision rules (alpha = 1e-2 primary)

Fold-level values average the three seeds within a fold. Tests are two-sided Wilcoxon signed-rank tests on the 10 fold pairs.

- **E1 (Q1, primary).** Test Pd at the texp64 threshold, G7 versus G3.
  - **Supported** if G7 > G3 in >= 8 of 10 folds and p < 0.05.
  - **Partial** if the mean difference is > 0 and p < 0.10.
  - **Not supported** otherwise.
- **E2 (Q2, primary).** Number of fold-seed runs with test Pfa > 2 alpha, out of 30.
  - **Supported** if G7 has <= 3 of 30 and does not exceed G3 by more than 2.
  - The mean |log10(Pfa / alpha)| for both arms is also reported.
- **E3 (Q3).** On IPIX1998, the three-seed mean Pd of G7 is greater than that of G3, the G7 test Pfa is <= 2 alpha in all three seeds, and the per-record G7 - G3 sign is reported.
  - **Supported** only if all of these hold.
- **Secondary (reported regardless of outcome):**
  - E1 and E2 at alpha = 1e-3;
  - adjacent-cell Pfa;
  - oracle Pd at a realized test Pfa of 1e-2 and 1e-3;
  - folds 1 and 6 separately (unseen target cell);
  - Pd by per-frame SCR proxy bin;
  - the D7 / P7 / ablation contrasts against G7, with the same Wilcoxon test.
- **No claims of the following kind:**
  - SA-RGB superiority over D7, unless E1-style criteria are met for G7 versus D7;
  - calibrated Neyman-Pearson Pfa (the paper reports an empirical rate on H0 cells of target frames);
  - performance at 1e-3 beyond descriptive reporting (about 13 expected events per fold).

## 8. Reporting

- Every arm, fold, seed and endpoint is reported, whatever the result.
- Negative or partial outcomes are reported as such, and the manuscript is written from the outcomes.
- Outputs go to `phaseC/` with per-run JSON, per-frame score arrays, and the exact config hash.

## Sign-off
- Author: Shiji Fan (chat approval). Date: 2026-10-08.
- Frozen code hash: written to `phaseC/runs_loro/config_hash.txt` at launch.

## Amendment log
(none)
