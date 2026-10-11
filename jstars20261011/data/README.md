# RadarGrid-CS research data

These are author-computed features, not raw radar echoes.

- `raw_features_0p512.npz`: IPIX1993 descriptors, ten records, four polarizations, 256 frames per record/polarization, 14 cells and seven features. Despite its historical filename, it contains no raw I/Q.
- `ipix1998_7feat.npz`: full-record and window-local IPIX1998 descriptors.
- Adjacent JSON manifests identify the input records and extraction metadata.
- Storage order: Hurst, RAA, RDPH, RVE, RI, NR, MCS. Slot order is defined in the supplied code.
- Predictions and table data: `../results/`; model weights: `../checkpoints/`.

Original radar data: McMaster IPIX collection, https://soma.mcmaster.ca/ipix/
Processing documentation: https://soma.mcmaster.ca/ipix/dartmouth/cdfhowto.html

Obtain source recordings from the provider and follow its reuse conditions. This repository does not redistribute them or claim ownership. The official site can be intermittently unavailable.

A new study should fit normalization on its training records and preserve record-level splits. IPIX1998 windows overlap and are not independent records.

Contact: shiji.fan@ua.pt.
