#!/usr/bin/env python3
"""Pose-only inference for a custom recording using the official EgoEMG loader.

No labels, guessed channel remapping, amplitude correction, or joint clipping.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_starts(n, window, left, right):
    valid_length = window - left - right
    if n < window or valid_length <= 0 or min(left, right) < 0:
        raise ValueError("Invalid recording/window/context dimensions")
    starts = list(range(0, n - window + 1, valid_length))
    if starts[-1] != n - window:
        starts.append(n - window)
    return np.asarray(starts, dtype=np.int64)


def merge_windows(aligned, starts, n, left, right, window):
    """Combine BCT windows; missing recording samples remain NaN."""
    expected = window - left - right
    if aligned.shape != (len(starts), 22, expected):
        raise ValueError("Unexpected aligned output shape")
    if not np.isfinite(aligned).all():
        raise ValueError("Model output contains nonfinite values")
    total = np.zeros((n, 22), dtype=np.float64)
    count = np.zeros(n, dtype=np.int32)
    for pred, start in zip(aligned, starts):
        lo, hi = int(start + left), int(start + window - right)
        if lo < 0 or hi > n:
            raise ValueError("Window outside recording")
        total[lo:hi] += pred.T
        count[lo:hi] += 1
    result = np.full((n, 22), np.nan, dtype=np.float32)
    good = count > 0
    result[good] = (total[good] / count[good, None]).astype(np.float32)
    return result, count


def video_indices(n, fs, fps):
    frame_time = np.arange(int(np.ceil(n / fs * fps)), dtype=np.float64) / fps
    indices = np.minimum(np.floor(frame_time * fs).astype(np.int64), n - 1)
    return frame_time, indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--preprocessing-metadata", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--egoemg-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    sys.path.insert(0, str(args.egoemg_root.resolve()))
    import torch
    from egoemg.checkpoints import load_branch
    from egoemg.constants import JOINTS
    from egoemg.kinematics import _batched_forward_kinematics, load_default_hand_model

    torch.manual_seed(0)
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    names = [j.name for j in JOINTS] + ["WRIST_FE", "WRIST_DEVIATION"]
    data = np.load(args.input, allow_pickle=False)
    meta = json.loads(args.preprocessing_metadata.read_text())
    fs = int(np.asarray(data["fs_hz"]).item())
    x = data["normalized_emg"].astype(np.float32)
    if fs != 2000 or x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all():
        raise ValueError("Expected finite N x 8 recording at 2000 Hz")
    if meta["hand"] != "left":
        raise ValueError("This run requires left-hand normalization")
    independent = ((data["filtered_emg_mv"] - data["norm_mean_mv"])
                   / (data["norm_std_mv"] + 1e-6)).astype(np.float32)
    np.testing.assert_allclose(x, independent, rtol=1e-6, atol=1e-6)
    model = load_branch(args.checkpoint).to(args.device).eval()
    if model.out_channels != 22 or model.provide_initial_pos:
        raise ValueError("Expected 22-angle pose-free EgoEMG checkpoint")
    left, right = model.left_context, model.right_context
    window, n = 8000, len(x)
    starts = window_starts(n, window, left, right)
    native, aligned = [], []
    with torch.inference_mode():
        for start in starts:
            batch = torch.from_numpy(x[start:start + window].T.copy())[None].to(args.device)
            pred = model({"emg": batch})
            if pred.shape[:2] != (1, 22) or not torch.isfinite(pred).all():
                raise RuntimeError("Unexpected model prediction")
            native.append(pred.cpu().numpy()[0])
            aligned.append(model.align_predictions(pred, window-left-right).cpu().numpy()[0])
        # Verify evaluation repeatability on the first window, without labels.
        again = model({"emg": torch.from_numpy(x[:window].T.copy())[None].to(args.device)})
        np.testing.assert_allclose(again.cpu().numpy()[0], native[0], rtol=1e-5, atol=1e-6)
    native = np.stack(native)
    aligned = np.stack(aligned)
    angles, coverage = merge_windows(aligned, starts, n, left, right, window)
    valid = coverage > 0
    assert valid[left:n-right if right else n].all() and not valid[:left].any()
    caution = data["sample_boundary_caution"].astype(bool)
    frame_time, indices = video_indices(n, fs, 30)
    frame_valid = valid[indices]
    landmarks = np.full((len(indices), 21, 3), np.nan, dtype=np.float32)
    hand = load_default_hand_model()
    with torch.inference_mode():
        # The vendored UmeTrack generic model is LEFT. Official viewer mirrors
        # x only for a right hand. Use all 22 angles (including wrist) here.
        neutral = _batched_forward_kinematics(torch.zeros(1, 22), hand)[0].numpy()
        rest = hand.landmark_rest_positions.numpy()
        np.testing.assert_allclose(neutral, rest, atol=1e-3)
        fk = _batched_forward_kinematics(torch.from_numpy(angles[indices[frame_valid]]), hand)
        landmarks[frame_valid] = fk.numpy()
    if not np.isfinite(landmarks[frame_valid]).all():
        raise ValueError("Nonfinite forward kinematics")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    time = np.arange(n) / fs
    np.savez_compressed(out / "predictions.npz", time_s=time, angles_rad=angles,
        angles_deg=np.rad2deg(angles), valid=valid, coverage_count=coverage,
        boundary_caution=caution, joint_names=np.asarray(names),
        window_start_sample=starts, native_window_angles_rad=native,
        aligned_window_angles_rad=aligned, fs_hz=fs, left_context=left, right_context=right)
    np.savez_compressed(out / "pose_video_data.npz", time_s=frame_time,
        sample_indices=indices, valid=frame_valid, landmarks_mm=landmarks,
        angles_deg=np.rad2deg(angles[indices]), neutral_landmarks_mm=neutral,
        boundary_caution=caution[indices], source_boundary_s=data["source_offsets_samples"][1:-1]/fs)
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "valid", "coverage_count", "boundary_caution"]
                        + [j + "_rad" for j in names] + [j + "_deg" for j in names])
        for i in range(n):
            values = list(angles[i]) + list(np.rad2deg(angles[i]))
            writer.writerow([f"{time[i]:.4f}", int(valid[i]), int(coverage[i]), int(caution[i])]
                            + [f"{v:.8g}" if np.isfinite(v) else "" for v in values])
    commit = subprocess.check_output(["git", "-C", str(args.egoemg_root), "rev-parse", "HEAD"], text=True).strip()
    transitions = sorted(set(int(s+left) for s in starts[1:]) |
                         set(int(s+window-right) for s in starts[:-1]))
    transitions = [s for s in transitions if left < s < n-right]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "hand": "left",
        "model": "EgoEMG EMGFormer-S", "official_commit": commit,
        "checkpoint_file": args.checkpoint.name, "checkpoint_sha256": digest(args.checkpoint),
        "input_npz_sha256": digest(args.input), "script_sha256": digest(__file__),
        "sources": [{"name": Path(s["path"]).name, "sha256": s["sha256"], "samples": s["samples"]} for s in meta["inputs"]],
        "fs_hz": fs, "samples": n, "duration_s": n/fs,
        "window_samples": window, "window_starts": starts.tolist(),
        "left_context": left, "right_context": right, "native_output_shape": list(native.shape),
        "alignment": "Official linear interpolate, align_corners=False, then mean of overlapping valid windows",
        "native_token_time_mapping": "Each token j has align_corners=False bin center left_context + (j+0.5)*valid_length/token_count - 0.5 within its window; endpoints clamp during interpolation. This is label alignment, not causal availability.",
        "valid_samples": int(valid.sum()), "missing_initial_s": left/fs,
        "maximum_coverage": int(coverage.max()), "joint_names": names,
        "window_transitions": [{"sample": s, "time_s": s/fs,
            "max_angle_jump_deg": float(np.max(np.abs(np.rad2deg(angles[s]-angles[s-1]))))}
            for s in transitions],
        "video": {"fps": 30, "frames": len(indices), "duration_s": len(indices)/30,
                  "frame_sampling": "floor(frame_time_s * 2000)", "coordinate_system": "canonical left UmeTrack, mm, identity global wrist transform"},
        "software": {"torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda},
        "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU",
        "checks": {"strict_checkpoint_restore": True, "normalization_consistent": True,
                   "repeat_inference_consistent": True, "neutral_fk_matches_rest": True,
                   "finite_valid_predictions": True, "complete_valid_coverage_after_context": True},
        "normalization_abs_z_gt_5_fraction": np.mean(abs(x)>5,axis=0).tolist(),
        "angle_range_deg": {name:[float(np.rad2deg(angles[valid,j]).min()),float(np.rad2deg(angles[valid,j]).max())] for j,name in enumerate(names)},
        "assumptions": {"input_unit": "uV (unverified)", "channel_mapping": "source columns 1..8 unchanged (unverified)", "continuous_files": "assumed", "preprocessing": "approximate paper-frequency recipe; saved input already appears filtered"},
        "interpretation": "Exploratory unlabeled predictions; no measured accuracy. No amplitude correction, clipping, or temporal smoothing. Noncausal offline inference."
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({k:summary[k] for k in ["native_output_shape","valid_samples","missing_initial_s","window_starts","video","checks"]},indent=2))


if __name__ == "__main__":
    main()
