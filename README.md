# EgoEMG qualitative demo report

## [▶ Open the interactive GT vs Prediction Demo](https://lixiaoxuan2001.github.io/egoemg-demo-report/)

Switch between EMGFormer-S, ViT-S (RGB), and Fusion (RGB + EMG), and filter the two public demo sessions directly in the browser.

This static GitHub Pages site shows 10 curated examples from the two public EgoEMG demo sessions. Each example compares ground-truth joint-angle articulation with predictions from:

- EMGFormer-S
- ViT-S (RGB)
- Fusion (RGB + EMG)

The viewer includes the public demo RGB frame, a bilateral canonical skeleton overlay, and per-hand angle MAE in degrees.

## Synchronized bilateral clips

[Play the continuous 20-second example and four 4-second comparisons](https://lixiaoxuan2001.github.io/egoemg-demo-report/bilateral-clips/). Each MP4 shows the first-person RGB recording, fixed-view EMGFormer-S pose for both hands, and 8 filtered EMG channels per wrist with a shared playback cursor. The page identifies the official split and time in the demo session for each clip. See the alignment manifests for the [short clips](bilateral-clips/media/manifest.json) and the [continuous test clip](bilateral-clips/media/manifest-20s.json).

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

## Custom left-hand recording (2026-11-26)

[Open the complete prediction video and waveform comparisons](https://lixiaoxuan2001.github.io/egoemg-demo-report/custom-left-20261126/). This separate experiment uses a user-recorded 20.172 s, 8-channel left-wrist EMG stream, with no ground-truth pose labels. Predictions, MP4 video, plots, provenance, and reproducibility scripts are included; raw EMG and checkpoints are not published. See [experiment notes](custom-left-20261126/README.md), including the unit/channel assumptions and window-boundary discontinuities.
