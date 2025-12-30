## 使用说明

### 输入流程（INPUT）

waveform -> WaveProcessor.get_spec [features: PQMF + STFT 实/虚拼接, 形状 (B, 2*F, T)] -> ResampleModule.downsample (patch_size=8) -> LLM

### 输出流程（OUTPUT）

LLM -> estimator -> ResampleModule.upsample -> WaveProcessor.get_wave [reconstruct: ISTFT + PQMF] -> waveform

### 文件说明（FILES）

rfwave.py: 基于复数谱的 flow matching。
wave_processor.py: 对 waveform 与复数谱进行预处理/后处理。
resample.py: 对复数谱进行上采样/下采样，将连续 8 帧压缩为一个向量。