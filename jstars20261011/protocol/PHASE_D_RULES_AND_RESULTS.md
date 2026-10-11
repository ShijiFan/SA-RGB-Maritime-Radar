# Phase D: temporal information and observation duration (diagnosis, no network)

Written 2026-10-09 before any D1 result was computed.

## Question
The seven descriptors are computed inside one observation. Does information across consecutive
observations (target persistence versus transient sea spikes) add detection power, and how does
it interact with the observation duration d in {0.128, 0.256, 0.512, 1.024} s?

## Data and protocol
- IPIX1993, the four duration caches in `rev3/cache_e3` (all re-extracted from the same raw pulses).
- LORO folds as before (test k, validation k+1, record 9 for k = 10), all four polarizations.
- Evaluated frames: those ending at or after 8.192 s for every configuration, so all
  configurations cover the same time span.

## Scorer (position-free, deterministic)
- HGB (max_iter 200, lr 0.1, 31 leaves, balanced classes, random_state 0).
- Per cell and frame: 7 per-frame-normalized descriptors plus the range-ordered peak margin to
  the +-1 neighbors (14 dims).
- Temporal window K in {1, 2, 4, 8} causal frames: mean, std and min of the 14 dims over the
  window (42 dims); K = 1 uses the 14 dims only.
- Dwell T = K x d.

## Endpoints
- Oracle Pd at realized test Pfa 1e-2 and 1e-3 (separability).
- Fixed validation threshold at 1e-2: Pd, Pfa, folds above 2 alpha (transfer).

## Decision rules (fixed now)
- D-a (temporal gain): at d = 0.512 s, the best K > 1 raises oracle Pd at 1e-2 by at least
  0.05 over K = 1 in at least 8 of 10 folds. If not met, no temporal network is built.
- D-b (no transfer cost): the folds above 2 alpha at 1e-2 with the fixed threshold do not
  rise by more than 1 relative to K = 1.
- D-c (short observation, descriptive): Pd versus d at K = 1, and Pd at equal dwell T for
  (d, K) pairs with the same T, to see whether K short frames match one long frame.

## D1 result (2026-10-09, `D1_temporal.csv`, `d1_summary.py`)
- Oracle Pd at 1e-2 (rows d, columns K = 1/2/4/8): 0.128 s 0.606/0.671/0.731/0.784; 0.256 s 0.669/0.730/0.793/0.850;
  0.512 s 0.705/0.773/0.841/0.887; 1.024 s 0.732/0.804/0.872/0.911.
- D-a met: at 0.512 s, K = 4 gains +0.136 (8/10 folds >= 0.05; the other two folds are saturated at 0.97-1.00), K = 8 +0.183, Wilcoxon p = 0.002.
- D-b met: folds above 2 alpha with the fixed threshold 2 -> 3. However, mean Pfa/alpha rises from 1.59 to 1.95 as K grows (0.512 s), so transfer needs the tail rule.
- D-c: at equal dwell, K short frames match or exceed one long frame (T = 1.024 s: 4 x 0.256 s 0.793 vs 1 x 1.024 s 0.732).
- At 1e-3: 0.507 -> 0.702 (0.512 s, K = 4).
- K = 1 at 0.512 s (0.705) reproduces the earlier HGB oracle (0.700), which is a sanity check.

## D2: cross-site temporal gain and tail-rule control (rules written 2026-10-09 before D2 was run)
- Same HGB scorer and temporal features as D1, d = 0.512 s, K in {1, 2, 4, 8}. Scores are logits.
- IPIX1998: train on 1993 frames 0-180 of all records, calibrate on 1993 frames 181-225, and test on all 1998 windows (117 frames).
  Every K is evaluated on 1998 frames 7-116, so all K share the same frames.
- Rules: the fixed threshold and the tail CFAR (W = 64, 90th percentile, c calibrated at 1e-2), as in Phase C.
- X-a (cross-site gain): 1998 oracle Pd at 1e-2 for K = 4 exceeds K = 1 by at least 0.05, and the per-record oracle Pd is higher in at least 7 of 9 records.
- X-b (control with the tail rule, K = 4): IPIX1993 LORO folds above 2 alpha at 1e-2 are at most 3 of 10, and the 1998 realized Pfa is at most 2 alpha.
- 1e-3 is reported descriptively. K = 8 is reported but is not used for the rules.

## D2 result (`D2_crosssite.csv`)
- X-a met:
  - IPIX1998 oracle Pd at 1e-2 rises from 0.363 (K = 1) to 0.515 (K = 4) and 0.631 (K = 8).
  - K = 4 is higher than K = 1 in 9 of 9 records. R6 (0.12-0.13) and R8 (0.26-0.37) stay weak.
- X-b met (K = 4, tail rule):
  - IPIX1993 LORO: Pd 0.824, Pfa 1.02 alpha, 0 of 10 folds above 2 alpha.
  - IPIX1998: Pd 0.528 at 1.19 alpha.
  - With the fixed threshold, the realized Pfa grows with K (LORO 1.59 -> 1.79 alpha).
- Descriptive, 1e-3, K = 4, tail rule: LORO 0.83 alpha with 0/10 folds above 2 alpha; IPIX1998 3.0 alpha (the K = 1 design gives about 5.9 alpha).
- K = 8, tail rule: LORO 0.869 at 1.11 alpha; IPIX1998 0.672 at 1.39 alpha.
- Comparison with the manuscript (range-ordered network, IPIX1998): 0.429 at 1.69 alpha.

## N1: temporal RadarGrid-CS (design and rules written 2026-10-09 before any network run)
### Design
- Per frame: the SA-RGB G7 grid passes the parameter-free front end of the range-ordered design.
  That front end is the slot readout with anchor lift, SCN, and the +-1 range peak margin, giving 72 values per cell.
- Per-cell frame encoder phi, shared across cells and frames: 72 -> 64 -> 64, SiLU, dropout 0.1.
- Causal temporal pooling over the last K frames: mean, std and min of phi, giving 192 values.
- Head psi: 192 -> 64 -> 1. There is no cell index and no grid direction, so the model is range-equivariant.
- K = 4 (2.048 s) is primary and K = 8 is secondary. K = 1 is the same architecture without temporal information and serves as the control.
- Training follows Phase C HP: 40 epochs, AdamW lr 1e-3, wd 1e-4, batch 256, pos_weight 13, best validation BCE; seeds 0-2.
- Frames:
  - Train on frames >= 7 of the training sequences.
  - LORO: evaluate frames 15-255 of the validation and test records.
  - IPIX1998: train on 1993 frames 7-180, validate on 181-225, test on 1998 frames 7-116.
- Matched reference: HGB on the mean, std and min of the same 72-value tensor over the same K frames (216 values).
### Rules
- N-a (IPIX1993 LORO, alpha 1e-2, tail rule):
  - The fold-mean Pd of K = 4 exceeds K = 1 in at least 8 of 10 folds, with Wilcoxon p < 0.05.
  - K = 4 has at most 3 of 30 runs above 2 alpha.
- N-b (IPIX1998, tail rule): Pd of K = 4 exceeds K = 1, and the realized Pfa is at most 2 alpha in all 3 seeds.
- The network-versus-HGB comparison and K = 8 are reported descriptively.

## N1 result (`N1_temporal_net.csv`, `n1_summary.py`; 21,249 parameters)
- N-a met: IPIX1993 LORO, tail rule, alpha 1e-2.
  - K = 1: 0.685 at 1.13 alpha. K = 4: 0.829 at 1.08 alpha. K = 8: 0.872 at 1.16 alpha.
  - K = 4 beats K = 1 in 10 of 10 folds (p = 0.002), with 0 of 30 runs above 2 alpha.
  - At 1e-3 (descriptive), K = 4 gives 0.646 at 0.78 alpha with 1 of 30 runs above 2 alpha; K = 1 gives 1.88 alpha with 14 of 30.
- N-b met: IPIX1998, tail rule.
  - K = 4: 0.570 at 1.17 alpha (seeds 1.15-1.18 alpha). K = 1: 0.411 at 1.68 alpha.
  - At 1e-3: 2.5 alpha (K = 1: 4.9 alpha).
- Network versus HGB on the same tensor:
  - LORO: equal (network higher in 5 of 10 folds at every K).
  - IPIX1998, K = 4: network 0.570 at 1.17 alpha versus HGB 0.513 at 1.21 alpha; oracle 0.564 versus 0.501. HGB is a single deterministic run.
- K = 8 on IPIX1998 is seed-sensitive (0.438-0.606).
- Disclosure: D2 had already shown the temporal gain on IPIX1998 with HGB before N1 was run; N1 is the second look at IPIX1998 for the temporal idea.

## N2 result (`N2_duration.csv`; descriptive; 0.128 s dropped at the user's request because it is below the durations used in the literature)
- IPIX1993 LORO, tail rule, alpha 1e-2, Pd (Pfa/alpha):

  | d | K = 1 | K = 2 | K = 4 | K = 8 |
  |---|---|---|---|---|
  | 0.256 s | 0.654 (1.13) | 0.727 | 0.798 | 0.852 |
  | 0.512 s | 0.685 | 0.763 | 0.829 | 0.872 |
  | 1.024 s | 0.702 | 0.771 | 0.839 | 0.885 |

- Runs above 2 alpha: 0-2 of 30 everywhere, except 1.024 s with K = 8 (4 of 30).
- Equal dwell 1.024 s: 4 x 0.256 s gives 0.798, 2 x 0.512 s gives 0.763, 1 x 1.024 s gives 0.702.
- Equal dwell 2.048 s: 8 x 0.256 s gives 0.852, 4 x 0.512 s gives 0.829, 2 x 1.024 s gives 0.771.
- The 0.512-s rows reproduce N1 exactly (determinism check).

## Checks before deciding on manuscript use (criteria written 2026-10-09 before running)
### M (mechanism): network, d = 0.512 s, K in {1, 4}, seeds 0-2, rerun deterministically with scores saved
- Tail calibration ratio: realized test Pfa / alpha for alpha in {1e-1, 3e-2, 1e-2, 3e-3, 1e-3}, with c set at 1e-2.
- Tail-shape index per test record and seed, on the test H0 scores: h = (q0.999 - q0.99) / (q0.99 - q0.9).
  An exponential tail gives h = 1; h > 1 means a heavier tail.
- Location drift: |q0.99(test H0) - q0.99(val H0)| / beta_test, with beta_test = (q0.99 - q0.9) / ln 10.
- Supported if, with K = 4 relative to K = 1:
  - (i) the 1e-3 ratio is closer to 1 and h is closer to 1 (mean over runs);
  - (ii) the location drift is larger.
  - Only (i) holding gives a partial mechanism; only the confirmed parts are written.
### Q (confound): equal dwell with the number of training samples matched
- Training frames are subsampled so that every duration has the training-sample count of 1.024 s: stride 2 at 0.512 s, stride 4 at 0.256 s.
- Compare (0.256, 4), (0.512, 2) and (1.024, 1) at T = 1.024 s, and (0.256, 8), (0.512, 4) and (1.024, 2) at T = 2.048 s.
  Tail rule, alpha 1e-2, 3 seeds, LORO.
- The short-frame advantage counts as real if the 0.256-s configuration still exceeds the 1.024-s one at both T in at least 7 of 10 folds.

## Check results
### Q (`CHK_Q_confound.csv`): supported
Training samples were matched at 121 frames per sequence; tail rule, alpha 1e-2.

| Dwell | 0.256 s | 0.512 s | 1.024 s | 0.256 s beats 1.024 s |
|---|---|---|---|---|
| T = 1.024 s | 0.780 | 0.748 | 0.702 | 10/10 folds |
| T = 2.048 s | 0.834 | 0.815 | 0.771 | 10/10 folds |

- Against 0.512 s, the 0.256-s configuration wins 7/10 folds at T = 1.024 s and 8/10 at T = 2.048 s.
- Unmatched gaps were 0.798 and 0.852, so part of the advantage was sample count, but most of it remains.
- Realized Pfa is 1.04-1.12 alpha in all six configurations.

### M (`CHK_M_mechanism.csv`): only partly supported
**Calibration ratio** (realized Pfa / alpha, for alpha = 1e-1, 3e-2, 1e-2, 3e-3, 1e-3):

| Protocol | K = 1 | K = 4 |
|---|---|---|
| LORO | 0.74 / 0.95 / 1.13 / 1.42 / 1.88 | 0.90 / 1.11 / 1.08 / 0.88 / 0.78 |
| 1998 | 0.67 / 1.04 / 1.68 / 2.87 / 4.92 | 0.69 / 0.91 / 1.17 / 1.68 / 2.51 |

The calibration curve is flatter with K = 4 on both protocols.

**Tail-shape index h** (1 means exponential):

| Protocol | K = 1 | K = 4 | Reading |
|---|---|---|---|
| LORO | 0.91 | 0.75 | The global record tail is already lighter than exponential at K = 1 and gets lighter, so "closer to 1" fails on LORO. |
| 1998 | 1.37 | 1.18 | Closer to 1. |

**Location drift:**

| Protocol | K = 1 | K = 4 | Reading |
|---|---|---|---|
| LORO | 0.72 | 0.92 | Larger in 8/10 folds. |
| 1998 | 1.22 | 0.65 | Smaller, which contradicts (ii). The fixed threshold on 1998 also improved, 3.17 -> 2.01 alpha. |

**Conclusion:**
- The claim "averaging lightens a heavy score tail while location drift grows" is not consistently supported, so it is not written.
- Writable fact: the tail rule's realized-to-nominal ratio stays within 0.78-1.11 over alpha = 1e-1 to 1e-3 with K = 4 on IPIX1993, against 0.74-1.88 with K = 1. On IPIX1998 the 1e-3 ratio falls from 4.9 to 2.5.

## N3: final detector, G7 versus G3 (rule written 2026-10-09 before running)
- Final detector: range-ordered front end with temporal pooling (N1 design), K = 4, d = 0.512 s. Arms G7 and G3, seeds 0-2.
- Rule F-a: on IPIX1993 LORO with the tail rule at alpha 1e-2, the fold-mean Pd of G7 exceeds G3 in at least 8 of 10 folds, Wilcoxon p < 0.05.
- Descriptive:
  - IPIX1998 G7 versus G3 per record;
  - K = 1 rows for both arms;
  - scores and checkpoints saved;
  - latency of one forward pass.

## N3 result (`N3_final.csv`, `final/`, `N3_latency.json`)
- **Model:** 21,249 parameters; one frame update takes 0.23 ms on CPU and 0.22 ms on GPU (median).
- **F-a NOT met.** IPIX1993 LORO, tail rule, 1e-2, K = 4: G7 0.829 at 1.08 alpha versus G3 0.787 at 1.05 alpha. G7 is higher in 7 of 10 folds (p = 0.375). At K = 1: G7 0.685 versus G3 0.541, 10 of 10 folds, p = 0.002.
- **Seven descriptors keep the advantage at a strict budget** (1e-3, K = 4): G7 0.646 at 0.78 alpha with 1 of 30 runs above 2 alpha; G3 0.637 at 1.80 alpha with 13 of 30.
- **IPIX1998, K = 4, tail rule:**
  - At 1e-2: G7 0.570 at 1.17 alpha versus G3 0.438 at 1.10 alpha. Oracle: 0.564 versus 0.443. G7 is higher in 8 of 9 records; R8 is tied (-0.0005). Against G7 with K = 1, K = 4 is higher in 7 of 9 records (R6 and R8 are lower).
  - At 1e-3: G7 0.341 (2.5 alpha) versus G3 0.247 (2.9 alpha).
- **Reading:** within one campaign, accumulation substitutes for part of the descriptor expansion. Across sites and at the strict budget, the seven descriptors keep their advantage.

## N4 (`N4_shift_final.csv`): the final detector is shift-insensitive. At s = 0 / +-1 / +-2, Pd is 0.829 / 0.827 / 0.825 and Pfa is 1.08 / 1.11 / 1.09 alpha. Fold 6 reaches 0.798.

## Round-2 review controls (rules written 2026-10-09 before running)
### F (fair equal dwell), primary comparison at a dwell of 1.024 s: (d = 0.256 s, K = 4) versus (d = 1.024 s, K = 1); (0.512 s, K = 2) is also reported. Same final architecture, G7, seeds 0-2, LORO.
- Decision times: common end times t_j = (j + 1) x 1.024 s. The window of K native frames ending at native index n_j = (j + 1) s - 1 (s = 1.024/d) covers exactly the same pulses for every d.
- Training: j = 7..127 (121 windows per sequence, the same pulses for all d).
- Background: the tail rule at native frame rate with the same physical history of 32.768 s, i.e., W = 128 / 64 / 32 native frames. Decisions are scored only at the common times j = 40..127. c is recalibrated on the validation decisions at 1e-2.
- Rule F-a: the fold-mean Pd of (0.256, 4) exceeds (1.024, 1) in at least 8 of 10 folds with Wilcoxon p < 0.05, and both mean realized Pfa are at most 2 alpha.
- Descriptive: oracle Pd at equal realized Pfa = 1e-2, and dwell 2.048 s with (0.256, 8) vs (0.512, 4) vs (1.024, 2) on the same frames.
- If F-a fails, the short-observation claim is restricted to the operating configuration and the causal attribution is dropped.

### S (simple temporal mean, no training): G7, d = 0.512 s.
- Single-frame logits (K = 1 network), the mean of four consecutive single-frame logits, and the trained K = 4 network, all from the saved scores.
- Common decision frames: 1993 frames 19-255 and 1998 frames 11-116, after truncating every score series to start at frame 18 / 10 so that the tail history is identical.
- c is recalibrated on validation for each method. Reported: tail-rule Pd, Pfa/alpha, runs above 2 alpha at 1e-2 and 1e-3, and oracle Pd at equal realized Pfa.
- No rule: this is a control that decides the wording. If the mean matches the network at equal Pfa, the contribution is temporal accumulation itself and the network is credited only for what it adds in calibration.
