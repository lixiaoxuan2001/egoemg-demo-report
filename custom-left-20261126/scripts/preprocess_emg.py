#!/usr/bin/env python3
"""Reproducible, explicitly approximate EgoEMG preprocessing of saved MAT signals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
from scipy.io import loadmat, savemat

FS = 2000
WINDOW = 8000
PADLEN = 2000
MARGIN = 1000
NOTCHES = list(range(50, 451, 50))
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design_sos():
    sections = [signal.butter(4, 20, "highpass", fs=FS, output="sos"),
                signal.butter(4, 450, "lowpass", fs=FS, output="sos")]
    for frequency in NOTCHES:
        b, a = signal.iirnotch(frequency, Q=30, fs=FS)
        sections.append(signal.tf2sos(b, a))
    return np.vstack(sections)


def filter_emg(values, sos):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or len(values) <= PADLEN:
        raise ValueError("EMG must have shape (N, 8), with N > 2000")
    if not np.isfinite(values).all():
        raise ValueError("EMG contains nonfinite values")
    return signal.sosfiltfilt(sos, values, axis=0, padtype="odd", padlen=PADLEN)


def make_windows(normalized, caution):
    if len(normalized) < WINDOW:
        raise ValueError("At least 8000 samples are required for one complete network window")
    starts = np.arange(0, len(normalized) - WINDOW + 1, WINDOW, dtype=np.int64)
    windows = np.stack([normalized[s:s + WINDOW].T for s in starts]).astype(np.float32)
    flags = np.array([caution[s:s + WINDOW].any() for s in starts], dtype=bool)
    return windows, starts, flags


def validate_filter(sos):
    """Independent tones test the implemented double-pass amplitude and phase."""
    t = np.arange(FS * 12) / FS
    checks = []
    for frequency in [10, 20, 80, 275, 450, 700, *NOTCHES[:-1]]:
        x = np.sin(2 * np.pi * frequency * t)
        y = filter_emg(np.tile(x[:, None], (1, 8)), sos)[:, 0]
        sl = slice(FS * 3, -FS * 3)
        basis = np.column_stack([np.sin(2 * np.pi * frequency * t[sl]),
                                 np.cos(2 * np.pi * frequency * t[sl])])
        coeff = np.linalg.lstsq(basis, y[sl], rcond=None)[0]
        measured = float(np.linalg.norm(coeff))
        _, h = signal.sosfreqz(sos, worN=[frequency], fs=FS)
        expected = float(abs(h[0]) ** 2)
        if frequency in NOTCHES:
            assert measured < 1e-5, (frequency, measured)
        else:
            assert abs(measured - expected) < 1e-4, (frequency, measured, expected)
        if frequency in [80, 275]:
            assert abs(np.arctan2(coeff[1], coeff[0])) < 1e-4
        checks.append({"frequency_hz": frequency, "measured_gain": measured,
                       "expected_double_pass_gain": expected})
    return checks


def comparison_plot(raw, filtered, time, lo, hi, boundaries, path, title):
    fig, axes = plt.subplots(8, 2, figsize=(14, 12), sharex=True, layout="constrained")
    for channel in range(8):
        limit = max(float(np.max(abs(raw[lo:hi, channel]))),
                    float(np.max(abs(filtered[lo:hi, channel]))), 1e-6) * 1.05
        for col, (data, color) in enumerate([(raw, "#586f8b"), (filtered, "#007f78")]):
            ax = axes[channel, col]
            ax.plot(time[lo:hi], data[lo:hi, channel], color=color, lw=.6)
            ax.set_ylim(-limit, limit)
            for boundary in boundaries:
                a, b = max(time[lo], (boundary-MARGIN)/FS), min(time[hi-1], (boundary+MARGIN)/FS)
                if a < b:
                    ax.axvspan(a, b, color="#f6b54c", alpha=.13)
                if time[lo] <= boundary/FS <= time[hi-1]:
                    ax.axvline(boundary/FS, color="#bd6a13", lw=.8, ls="--")
            if col == 0:
                ax.set_ylabel(f"CH {channel+1}\nµV*")
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Before: saved signal")
    axes[0, 1].set_title("After: approximate preprocessing")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s); amber = boundary review region")
    fig.supxlabel("*Input unit assumed; both columns use identical scales per channel", fontsize=10)
    fig.suptitle(title, fontsize=15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def spectra(raw, filtered, offsets):
    results = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if hi-lo-2*MARGIN < 4096:
            raise ValueError("Each recording needs at least 6096 samples for fixed PSD analysis")
        kwargs = dict(fs=FS, window="hann", nperseg=4096, noverlap=2048,
                      detrend="constant", scaling="density", axis=0)
        f, before = signal.welch(raw[lo+MARGIN:hi-MARGIN], **kwargs)
        _, after = signal.welch(filtered[lo+MARGIN:hi-MARGIN], **kwargs)
        results.append((f, before, after))
    return results


def spectrum_plot(results, names, path):
    fig, axes = plt.subplots(8, len(results), figsize=(7*len(results), 13),
                             sharex=True, squeeze=False, layout="constrained")
    for ch in range(8):
        maximum = max(float(np.max(r[1][:, ch])) for r in results)
        upper = 10*np.log10(max(maximum, 1e-30)) + 5
        for j, (f, before, after) in enumerate(results):
            ax = axes[ch, j]
            for values, color, label in [(before, "#8795a5", "Before"), (after, "#007f78", "After")]:
                ax.plot(f, 10*np.log10(np.maximum(values[:, ch], 1e-30)), lw=.7, color=color, label=label)
            ax.set(xlim=(0, 1000), ylim=(upper-130, upper))
            for hz in [20, 450]:
                ax.axvline(hz, color="#bd6a13", ls="--", lw=.6)
            ax.grid(alpha=.15)
            if j == 0:
                ax.set_ylabel(f"CH {ch+1}\ndB re 1 µV²/Hz")
            if ch == 0:
                ax.set_title(names[j]); ax.legend(fontsize=8)
            if ch == 7:
                ax.set_xlabel("Frequency (Hz)")
    fig.suptitle("Before / after PSD | same absolute scale per channel\nWelch 4096 / 2048; each file's first and last 0.5 s excluded", fontsize=14)
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="MAT files in acquisition order; order is preserved")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-unit", choices=["uV", "mV"], default="uV")
    parser.add_argument("--stats", type=Path, default=ROOT / "assets/egoemg_emg_normalization_filtered.json")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    output_names = ["processed_emg.npz", "processed_emg.mat", "metadata.json", "metrics.csv",
                    "waveforms_full.png", "waveforms_zoom.png", "psd_comparison.png", "filter_response.png", "report.md"]
    if any(p.resolve() in [out/n for n in output_names] for p in args.inputs):
        raise ValueError("Output would overwrite an input")
    hashes = [sha256(p) for p in args.inputs]
    arrays = []
    for p in args.inputs:
        data = loadmat(p)
        x = data.get("rawdata_save")
        if x is None or np.iscomplexobj(x) or x.ndim != 2 or x.shape[1] != 8:
            raise ValueError(f"Invalid rawdata_save: {p}")
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite input: {p}")
        arrays.append(x.astype(np.float64))
    original = np.concatenate(arrays)
    raw_uv = original * (1000 if args.input_unit == "mV" else 1)
    offsets = np.r_[0, np.cumsum([len(x) for x in arrays])]
    n = len(original)
    time = np.arange(n, dtype=np.float64) / FS
    caution = np.zeros(n, dtype=bool)
    for b in offsets:
        caution[max(0, b-MARGIN):min(n, b+MARGIN)] = True
    sos = design_sos()
    filtered = filter_emg(raw_uv, sos)
    stats_all = json.loads(args.stats.read_text())
    stats = stats_all["emg_filtered_left"]
    mean, std = np.array(stats["mean"]), np.array(stats["std"])
    if stats["unit"] != "mV" or mean.shape != (8,) or std.shape != (8,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid official left-hand normalization statistics")
    filtered_mv = filtered / 1000
    normalized = ((filtered_mv-mean)/(std+1e-6)).astype(np.float32)
    windows, starts, flags = make_windows(normalized, caution)
    tail_start = len(windows)*WINDOW
    payload = dict(saved_emg=original, saved_emg_uv=raw_uv, filtered_emg_uv=filtered,
                   filtered_emg_mv=filtered_mv, normalized_emg=normalized,
                   network_windows=windows, time_s=time, fs_hz=np.int64(FS),
                   window_start_sample=starts, window_start_time_s=starts/FS,
                   window_boundary_caution=flags, sample_boundary_caution=caution,
                   source_offsets_samples=offsets, tail_start_sample=np.int64(tail_start),
                   normalized_tail=normalized[tail_start:], filter_sos=sos,
                   norm_mean_mv=mean, norm_std_mv=std)
    assert all(np.isfinite(a).all() for a in payload.values())
    np.savez_compressed(out/"processed_emg.npz", **payload)
    savemat(out/"processed_emg.mat", payload, do_compression=True, oned_as="column")
    # Reopen both formats; exact comparison also verifies window orientation.
    with np.load(out/"processed_emg.npz") as saved:
        for key, value in payload.items():
            np.testing.assert_array_equal(saved[key], value)
    mat = loadmat(out/"processed_emg.mat")
    for key, value in payload.items():
        np.testing.assert_array_equal(mat[key].reshape(np.asarray(value).shape), value)
    np.testing.assert_array_equal(windows[0], normalized[:WINDOW].T)
    for index in [0, n//2, n-1]:
        np.testing.assert_allclose(normalized[index], (filtered[index]/1000-mean)/(std+1e-6), rtol=1e-6, atol=1e-7)
    results = spectra(raw_uv, filtered, offsets)
    metrics = []
    for j, (f, before, after) in enumerate(results):
        lo, hi = offsets[j]+MARGIN, offsets[j+1]-MARGIN
        for ch in range(8):
            row = {"file": args.inputs[j].name, "channel": ch+1,
                   "scope": "per_file_excluding_first_last_0.5s"}
            for label, x, psd in [("before", raw_uv, before), ("after", filtered, after)]:
                row[f"{label}_rms_uv"] = float(np.sqrt(np.mean(x[lo:hi, ch]**2)))
                row[f"{label}_peak_abs_uv"] = float(np.max(abs(x[lo:hi, ch])))
                for name, mask in [("lt20", f<20), ("20to450", (f>=20)&(f<=450)), ("gt450", f>450)]:
                    power = float(np.sum(psd[mask, ch])*(f[1]-f[0]))
                    row[f"{label}_{name}_power_uv2"] = power
                    row[f"{label}_{name}_fraction"] = float(np.sum(psd[mask, ch])/max(np.sum(psd[:, ch]), 1e-300))
            metrics.append(row)
    with (out/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    comparison_plot(raw_uv, filtered, time, 0, n, offsets, out/"waveforms_full.png",
                    "Left wrist EMG | 2000 Hz | full recording")
    duration = 2*FS
    energy = np.sum(filtered**2, axis=1)
    prefix = np.r_[0., np.cumsum(energy)]
    bad = np.r_[0, np.cumsum(caution)]
    candidates = np.arange(n-duration+1)
    valid = bad[candidates+duration] == bad[candidates]
    if not np.any(valid):
        raise ValueError("No complete 2 s interval outside boundary review regions")
    candidates = candidates[valid]
    zoom_start = int(candidates[np.argmax(prefix[candidates+duration]-prefix[candidates])])
    comparison_plot(raw_uv, filtered, time, zoom_start, zoom_start+duration, offsets,
                    out/"waveforms_zoom.png", "Highest-energy 2 s interval outside boundary review regions")
    for j, b in enumerate(offsets[1:-1], 1):
        comparison_plot(raw_uv, filtered, time, b-MARGIN, b+MARGIN, offsets,
                        out/f"waveforms_boundary_{j}.png", f"File boundary at {b/FS:.3f} s | continuity assumed")
    spectrum_plot(results, [p.stem[-6:] for p in args.inputs], out/"psd_comparison.png")
    freq, h = signal.sosfreqz(sos, worN=32768, fs=FS)
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    ax.plot(freq, 20*np.log10(np.maximum(abs(h), 1e-12)), label="Single pass", alpha=.65)
    ax.plot(freq, 40*np.log10(np.maximum(abs(h), 1e-12)), label="Actual forward-backward magnitude")
    ax.set(xlim=(0, 1000), ylim=(-120, 3), xlabel="Frequency (Hz)", ylabel="Gain (dB)", title="Designed frequency response (interior steady-state)")
    ax.legend(); ax.grid(alpha=.2); fig.savefig(out/"filter_response.png", dpi=160); plt.close(fig)
    validation = validate_filter(sos)
    assert hashes == [sha256(p) for p in args.inputs], "Input files changed"
    metadata = dict(created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        description="Approximate paper-frequency recipe applied to already processed saved EMG",
        hand="left", fs_hz=FS, input_unit_assumption=args.input_unit, unit_verified=False,
        channel_mapping_verified=False, channel_order="preserved; source columns 1 through 8",
        continuity_assumed=True, original_samples=n, duration_s=n/FS,
        inputs=[dict(path=str(p.resolve()), sha256=hsh, samples=len(x)) for p,hsh,x in zip(args.inputs, hashes, arrays)],
        filter=dict(highpass_hz=20, lowpass_hz=450, butterworth_order_each=4,
                    notch_hz=NOTCHES, notch_q=30, notch_order_each=2, zero_phase=True,
                    method="combined SOS sosfiltfilt", padtype="odd", padlen=PADLEN,
                    note="Forward-backward magnitude is single-pass magnitude squared; parameters are engineering defaults"),
        normalization=dict(statistics=stats, stats_sha256=sha256(args.stats), epsilon_mv=1e-6,
                           official_repository_commit="6b6e4172b7141d78eda9983a919e6dcbf3a754df"),
        windows=dict(shape=list(windows.shape), dtype="float32", layout="batch,channel,time",
                     starts_samples=starts.tolist(), boundary_caution=flags.tolist(), tail_samples=n-tail_start,
                     status="candidate inputs; unit and channel mapping require confirmation; no inference performed"),
        caution=dict(margin_samples=MARGIN, note="Advisory, not a measured transient duration or validity guarantee"),
        psd=dict(fs=FS,nperseg=4096,noverlap=2048,window="hann",detrend="constant",scaling="density",trim_each_file_samples=MARGIN),
        zoom_interval_s=[zoom_start/FS,(zoom_start+duration)/FS],
        normalized_abs_gt5_fraction=np.mean(abs(normalized)>5,axis=0).tolist(),
        validation=dict(tone_tests=validation, npz_mat_roundtrip="passed", normalization_reference="passed", original_hashes_unchanged=True),
        software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
        script_sha256=sha256(__file__))
    (out/"metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    ch2 = [r for r in metrics if r["channel"] == 2]
    rows = "\n".join(f"| {r['file']} | {r['before_gt450_fraction']:.2%} | {r['after_gt450_fraction']:.2%} | {r['before_rms_uv']:.3f} | {r['after_rms_uv']:.3f} |" for r in ch2)
    normalization_rows = "\n".join(f"| CH {i+1} | {v:.2%} |" for i,v in enumerate(metadata['normalized_abs_gt5_fraction']))
    report = f"""# 左手 EMG 预处理与对比

完成近似论文频率要求的零相位预处理，输入 {n} × 8、{n/FS:.3f} 秒；原始文件哈希未改变。

## 处理与限制

四阶 Butterworth 20 Hz 高通、四阶 450 Hz 低通，50 至 450 Hz 每隔 50 Hz 二阶陷波（Q=30）。组合 SOS 后双向滤波，odd 延拓 {PADLEN} 点。双向幅频响应为单向的平方，截止处衰减与单向不同；450 Hz 同时有陷波。阶数、Q 和延拓是工程选择，不是已确认的作者参数。

输入为软件已保存信号，已有高通/谐波抑制迹象。本次仍按选定流程再次滤波；无法恢复此前丢失的频率信息。假设输入单位为 {args.input_unit}，保持原通道顺序，假设两份数据连续。单位和官方通道映射均待确认。边界两侧 0.5 秒仅提示人工检查，不代表其外没有瞬态。

## 对比结果

第二通道的高频占比与 RMS（每份去掉首尾 0.5 秒；RMS 单位为假定 µV）：

| 文件 | 处理前 >450 Hz 功率占比 | 处理后占比 | 处理前 RMS | 处理后 RMS |
|---|---:|---:|---:|---:|
{rows}

功率下降是滤波的预期效果，不能直接解释为信噪比或姿态预测准确率提高。完整逐通道数值见 metrics.csv。频谱使用共同绝对单位，没有分别归一到各自峰值；图中低于显示下限的曲线只是被裁切。

## 图像

- [全段波形](waveforms_full.png)
- [局部波形：{zoom_start/FS:.3f}–{(zoom_start+duration)/FS:.3f} 秒](waveforms_zoom.png)
- [逐通道、逐文件频谱](psd_comparison.png)
- [滤波器单向与双向响应](filter_response.png)
- 文件交界波形见 waveforms_boundary_*.png；虚线为交界，浅橙区域为检查范围。

## 数据格式与后续使用

processed_emg.mat 与 processed_emg.npz 内容一致。saved_emg 保留原始数值；saved_emg_uv / filtered_emg_uv 用于物理波形对比；filtered_emg_mv 用于官方归一化；normalized_emg 为无量纲 float32。

network_windows 形状为 {list(windows.shape)}，顺序为 [批量, 通道, 时间]，每窗4秒。使用官方左手 mV 统计量：z=(filtered_emg_mv-mean)/(std+1e-6)。末尾 {n-tail_start} 点保存在 normalized_tail，不补零。时间和索引均从0开始；source_offsets_samples 使用左闭右开边界。窗口边缘提示为 {flags.tolist()}，不会自动删除任何窗口。

这些是候选网络输入，尚未核对佩戴映射与单位，也未运行网络。归一化后较多样本绝对值超过5，说明按当前单位假设，幅度与官方训练统计有明显差异，需要核对采集增益和单位后再解释网络输出。该差异也可能与采集动作和瞬态有关，不能仅凭幅度确定原因。未做额外裁剪或自适应归一化。

| 通道 | 全段归一化后绝对值 >5 的样本比例 |
|---|---:|
{normalization_rows}

## 验证

通过合成正弦波增益、通带相位、所有陷波、MAT/NPZ逐字段读回、归一化独立复算和原始文件哈希检查。所有输出有限。验证参数和结果详见 metadata.json。
"""
    (out/"report.md").write_text(report)
    print(json.dumps({"output": str(out), "samples": n, "windows": list(windows.shape),
                      "tail": n-tail_start, "ch2_metrics": ch2, "validation": "passed"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
