# EgoEMG qualitative demo report

This static GitHub Pages site shows 10 curated examples from the two public EgoEMG demo sessions. Each example compares ground-truth joint-angle articulation with predictions from:

- EMGFormer-S
- ViT-S (RGB)
- Fusion (RGB + EMG)

The viewer includes the public demo RGB frame, a bilateral canonical skeleton overlay, and per-hand angle MAE in degrees.

## Interpretation

These are functional examples, not benchmark results. Five eligible frames were selected at evenly spaced positions from each demo session, independently of model error.

The skeleton is a canonical, wrist-normalized articulation render produced from predicted or ground-truth joint angles. The models do not predict global wrist position, so the visualization must not be interpreted as full camera-space 3D hand reconstruction.

## Reproducibility

- Official code commit: `6b6e4172b7141d78eda9983a919e6dcbf3a754df`
- Samples: five from `sess_20260404_153935` and five from `sess_20261016_161026`
- Exact checkpoint SHA-256 values, camera frame IDs, and metrics: [`manifest.json`](manifest.json)
- Article: [arXiv:2605.05712](https://arxiv.org/abs/2605.05712)

No checkpoints, raw EMG streams, private filesystem paths, SSH details, or credentials are included.

## License and attribution

EgoEMG-authored demo recordings and annotations are licensed under CC BY-NC 4.0. The images and derived articulation renders in this repository are a noncommercial adaptation and attribute the EgoEMG authors. See [`LICENSE.md`](LICENSE.md).
