# SA-RGB research: current and legacy implementations

## Current manuscript: RadarGrid-CS

**Short-Observation Seven-Feature Detection of Small Floating Targets in Sea Clutter Under Record-Disjoint Evaluation**

Authors: Shiji Fan, Daniel Albuquerque, Feng Luo, Pedro Pinho, and Ming Shen.

The manuscript is prepared for JSTARS submission. This repository does not imply acceptance or publication.

- [Code and reproduction instructions](jstars20261011/README.md)
- [Derived feature data and provenance](jstars20261011/data/README.md)
- [Per-run predictions and experimental results](jstars20261011/results/)
- [Frozen model weights](jstars20261011/checkpoints/)
- [Supplementary tables and figures](jstars20261011/Supplementary_Material.pdf)

The snapshot contains the 21,249-parameter RadarGrid-CS detector with record-disjoint evaluation and causal tail-CFAR calibration. Its code and data must be used together; they differ from the older RadarGrid-Lite experiments.

Raw McMaster IPIX radar recordings are not redistributed. The data folder contains author-computed descriptor arrays and points to the original provider.

## Legacy implementation: RadarGrid-Lite and SAFR

The original root-level code and results are retained. [LEGACY_README.md](LEGACY_README.md) preserves the historical description, including its old title and status text. Those statements are historical and should not be used as the publication status or numerical results of the current manuscript.

## License and contact

The root MIT license applies to the authors' software. See the snapshot's [data notice](jstars20261011/DATA_NOTICE.txt) for non-code artifacts and third-party source data.

Shiji Fan: shiji.fan@ua.pt.
