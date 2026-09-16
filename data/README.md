# Radar Datasets & Data Preparation

This repository evaluates maritime small-target detection using public McMaster University IPIX radar datasets.

## 1. IPIX 1993 Dataset (Dartmouth, Nova Scotia)
- **Target**: Small spherical floating target (styrofoam ball wrapped in wire mesh, ~1m diameter) moored in open sea.
- **Radar Parameters**: X-band (9.39 GHz), Pulse Repetition Frequency (PRF) = 1000 Hz, Range resolution = 30 m.
- **Ten Standard Records**:
  - `19931107_135603` (Target at range cell 9)
  - `19931108_220902` (Target at range cell 7)
  - `19931109_191449` (Target at range cell 7)
  - `19931109_202217` (Target at range cell 7)
  - `19931110_001635` (Target at range cell 7)
  - `19931111_163625` (Target at range cell 8)
  - `19931118_023604` (Target at range cell 8)
  - `19931118_162155` (Target at range cell 7)
  - `19931118_162658` (Target at range cell 7)
  - `19931118_174259` (Target at range cell 7)
- **Polarizations**: HH, HV, VH, VV.

## 2. IPIX 1998 Dataset (Grimsby, Lake Ontario)
- **Target**: Controlled nautical boat / floating target.
- **Range Resolution**: 3m / 9m (high range resolution mode).
- Used for cross-year, cross-environment adaptation diagnostics.

## 3. Data Download
The raw `.mat` files are archived and publicly available through McMaster University:
- [IPIX Radar Database](http://soma.ece.mcmaster.ca/ipix/)

Place downloaded `.mat` files in `data/raw_ipix1993/`.

## 4. Synthesizing SA-RGB Dataset
To generate the slot-addressable RGB images:
```bash
python scripts/build_dataset.py \
    --source data/raw_ipix1993 \
    --output data/ipix1993_sargb_0p512 \
    --duration 0.512
```
