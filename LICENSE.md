# Licenses and attribution

## EgoEMG-derived media and results

The source RGB frames, annotations, and derived articulation visualizations under `assets/` in this repository are adapted from the EgoEMG public demo subset and are licensed under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC 4.0).

Attribution: EgoEMG authors, “EgoEMG,” arXiv:2605.05712.

- License: <https://creativecommons.org/licenses/by-nc/4.0/>
- Article: <https://arxiv.org/abs/2605.05712>

Changes made: ten frames were selected from the public demo sessions; RGB frames were resized/encoded for web delivery; official model checkpoints were used to generate predictions; joint angles were converted to canonical wrist-normalized skeleton images; and per-hand angle errors were calculated for display.

These materials are provided for noncommercial research and educational inspection. Third-party materials retain their respective licenses.

## Site source

The original HTML, CSS, and JavaScript used to present this report are available under the MIT License:

Copyright (c) 2026 lixiaoxuan2001

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Custom recording experiment

The separate `custom-left-20261126/` experiment uses a user-recorded EMG sequence, not recordings or annotations from the official EgoEMG demo dataset. It includes model-derived joint angles and canonical skeleton renders using the EgoEMG code and UmeTrack generic hand model. Attribution: EgoEMG authors and Meta Platforms, Inc. (UmeTrack). Third-party terms remain applicable; the provided EgoEMG and UmeTrack license texts are included in `custom-left-20261126/attribution/`. No model checkpoint or original EMG recording is redistributed.
