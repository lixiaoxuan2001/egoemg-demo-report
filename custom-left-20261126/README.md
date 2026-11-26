# 自采左手 EMG：完整预测、视频和波形

本实验使用 2026-11-26 自采的两段左手腕带数据，按文件顺序拼接为 40344 × 8，采样率 2000 Hz，时长 20.172 秒。使用官方 EgoEMG EMGFormer-S 权重离线推理；没有真实姿态标签，因此不报告准确率。

- [打开视频与交互曲线](index.html)
- [完整 1080p MP4](media/left_hand_pose.mp4)：30 fps、606 帧、20.2 秒，末尾按视频帧取整。
- [完整预测 CSV](data/predictions.csv)、[预测 NPZ](data/predictions.npz)、[运行记录](manifest.json)
- [预处理指标](data/metrics.csv)、[预处理参数](data/preprocessing_metadata.json)

## 输入、预处理与模型

输入文件为 `WL-EMG 20261126 172929.mat`（20168 点）及 `WL-EMG 20261126 172939.mat`（20176 点）。连续性由用户描述作假设，没有逐样本时间戳可独立验证。原始采集文件和完整 EMG 数组未发布。

保存信号已有高通和工频谐波抑制迹象；本次额外执行四阶 Butterworth 20 Hz 高通、四阶 450 Hz 低通，以及 50/100/150/200/250/300/350/400/450 Hz 二阶陷波（Q=30）。组合 SOS 双向滤波，odd 延拓 2000 点。阶数、Q 和延拓是工程选择，不是已确认的作者参数。前后波形使用相同纵轴尺度。

假设采集单位为 μV，除以 1000 转 mV，再使用官方左手统计量 `(x-mean)/(std+1e-6)`。保留原始 8 列通道顺序；未校验其与官方电极顺序的对应。没有额外缩放、裁剪或自适应归一化。各通道约 21%–82% 样本的归一化绝对值超过 5，不能将视频视为已验证的动作重建。

官方代码 commit 为 `6b6e4172b7141d78eda9983a919e6dcbf3a754df`。使用 EgoEMG 的 `emgformer_s.ckpt`，不是 EMG2Pose 的同类模型。checkpoint 按自身配置严格加载，不需要伪造姿态标签；`eval()` + `inference_mode()`，关闭 TF32，CUDA 推理。

## 时间轴、窗口与跳变

每窗 8000 点；起点 `[0,7490,14980,22470,29960,32344]`。模型左上下文 510 点、右上下文 0；每窗原始输出 `[22,150]`。沿用模型的 linear interpolate（align_corners=False）对齐到 7490 个有效样本，重叠处逐样本取平均，未施加额外时序平滑。原来预处理导出的 5 个不重叠候选窗口仅用于预处理检查；本次从完整 `normalized_emg` 重新取 6 个窗口，因此包含末尾 344 点。

前 0.255 秒无预测，NPZ 为 NaN，CSV 角度列留空；之后全部 39834 点有覆盖。2000 Hz 对齐结果是插值表示，不代表同等频率的独立模型预测。模型使用整窗信息，是非因果离线推理。

窗口变化时刻为 4.000、7.745、11.490、15.235、16.427、18.980 秒。实测部分交界出现明显跳变，最大约 45.8°；这是重要的窗口推理局限，不应直接解释为真实动作。逐交界数值见 manifest。视频在这些位置前后 0.1 秒显示提示，曲线以灰线标出。源文件接缝 10.084 秒用橙色标记，首尾及接缝前后 0.5 秒为输入检查区域；提示不等于其外没有异常。

## 预测与骨架约定

CSV 每行对应一个采集样本：`time_s`、`valid`、`coverage_count`、`boundary_caution`，随后 22 列 `_rad` 和 22 列 `_deg`。NPZ 包含相同数据及 `native_window_angles_rad[6,22,150]`、`aligned_window_angles_rad[6,22,7490]`、窗口起点与关节名称，全部可以 `allow_pickle=False` 读取。

前 20 个角度遵循官方 JOINTS 顺序，最后两列为 `WRIST_FE`、`WRIST_DEVIATION`。使用官方 UmeTrack 左手模板及全部 22 维角度生成 21 个骨架点，不镜像左手、不置零手腕角；零角度的正向运动学已与模板静止位置核对。图像是固定相机投影、固定尺度，骨架坐标为模板毫米坐标，不是个体化手型或相机空间位置。

视频按 `floor(frame_time_s * 2000)` 选择预测，前 8 帧无姿态；最后一帧时间 20.1667 秒，容器时长 20.2 秒。`pose_video_data.npz` 保存逐视频帧时间、采样索引、有效标记和预测骨架；交互曲线按相同视频帧采样。前后波形对比图中的输入单位仍是未验证的 μV 假设。

## 复现

需要自行提供原始 MAT 和官方权重；本仓库不分发这些文件。模型代码与环境安装遵循官方 EgoEMG 仓库。预处理需 NumPy/SciPy/Matplotlib；渲染另需 Pillow 和支持 libx264 的 ffmpeg。以下从本目录运行，示例路径请替换为自己的路径：

```sh
python scripts/preprocess_emg.py --inputs /path/to/first.mat /path/to/second.mat --output-dir work/preprocessing --input-unit uV
python scripts/infer_left_emg.py --input work/preprocessing/processed_emg.npz --preprocessing-metadata work/preprocessing/metadata.json --checkpoint /path/to/emgformer_s.ckpt --egoemg-root /path/to/EgoEMG --output-dir work/inference --device cuda:0
python scripts/render_left_emg.py --input-dir work/inference
python -m unittest discover -s scripts -p 'test_*.py' -v
```

也可直接复渲染已发布预测：建立空目录，复制 `data/predictions.npz`、`data/pose_video_data.npz` 和 `manifest.json` 到其中，再运行 `render_left_emg.py --input-dir` 指向该目录。默认保留初始缺失区间，渲染不会改变模型角度。

## 验证与归属

通过窗口覆盖、重叠平均、短输入、右上下文及视频帧取整测试；严格恢复权重、独立复算归一化、重复推理一致性、正向运动学静止位置、CSV/NPZ 内容和缺失标记均已检查。视频编码和关键帧、页面交互与下载链接另行验证。

模型与公开代码来源：[EgoEMG](https://github.com/zhenqis123/EgoEMG)。手模型与运动学来源：随 EgoEMG 分发的 UmeTrack。许可原文保存在 `attribution/`，第三方材料遵循各自许可。本页输入来自用户自采记录，不是官方公开数据集。网站原有官方示例的许可说明见根目录 LICENSE.md。
