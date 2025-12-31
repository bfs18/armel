# Poorman's AR-DiT TTS 📢

> **关键词**: ARDiT, AR-DiT, Autoregressive Diffusion Transformer, TTS, Text-to-Speech, Mel-Spectrogram

受 AR-DiT (ARDiT) 启发的**低资源友好**语音合成系统，采用自回归 Transformer（Qwen3 LLM）+ 扩散模型的架构，通过扩散过程生成 Mel 频谱，再经 Vocoder 转换为音频。

**✨ 最小实现的 AR-DiT TTS 训练推理 Pipeline**，可在单张 RTX 5090 (32GB) 上使用 8000 小时数据集，两天内训练出可懂的语音合成结果。

> **PS**: Diffusion backbone 使用的是 [RFWave](https://github.com/bfs18/rfwave) 的 ConvNeXt 架构，而非 DiT。

## 🌟 为什么选择本项目？

- 🚀 **低资源友好**：单卡 RTX 5090 (32GB) 即可处理 8000 小时数据集训练
- 📦 **最小实现**：代码简洁清晰，易于理解和修改，适合学习和二次开发
- 🇨🇳 **中文友好**：完整的中文文档和中文数据处理流程
- 🤗 **开箱即用**：提供预训练模型和处理好的数据集，快速上手
- 💡 **实用导向**：两天训练即可达到可懂效果，训练更久质量更好 - practical rather than perfect

## 🎵 生成示例

训练模型生成的音频示例：

<audio controls>
  <source src="outputs/inference_audio_203019_796b492db63e5ccaad85.wav" type="audio/wav">
  您的浏览器不支持音频播放。<a href="outputs/inference_audio_203019_796b492db63e5ccaad85.wav">下载音频</a>
</audio>

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 🤗 Hugging Face 资源

我们在 Hugging Face 上提供了预训练模型和训练数据集：

- **预训练模型**: [laupeng1989/armel-checkpoint](https://huggingface.co/laupeng1989/armel-checkpoint)
- **训练数据集**: [laupeng1989/armel-dataset](https://huggingface.co/datasets/laupeng1989/armel-dataset)

下载资源：
```bash
# 下载训练数据集
huggingface-cli download laupeng1989/armel-dataset --repo-type dataset --local-dir ./data/armel-dataset

# 下载预训练模型
huggingface-cli download laupeng1989/armel-checkpoint --local-dir ./models/armel-checkpoint
```

**💡 提示**：如果使用 Hugging Face 上的数据集，可以跳过下面的"数据准备"环节，直接进入训练步骤。

## 📊 数据准备

### 1️⃣ 准备原始数据

本项目使用 [Amphion Emilia 预处理器](https://github.com/open-mmlab/Amphion/tree/main/preprocessors/Emilia) 处理原始音频数据。

处理后的数据格式如下：
```
example_data/
├── 仙逆 第87集 身世苏醒（下） [638031163].json
├── 仙逆 第87集 身世苏醒（下） [638031163]_000000.m4a
├── 仙逆 第87集 身世苏醒（下） [638031163]_000001.m4a
├── 仙逆 第87集 身世苏醒（下） [638031163]_000002.m4a
└── ...
```

JSON 文件格式（包含分段信息和文本）：
```json
[
  {
    "duration": 10.94,
    "text": "[SPEAKER_00] 欢迎收听...",
    "speaker": 0,
    "parts": [
      {
        "text": "[SPEAKER_00] 欢迎收听...",
        "start": 4.5125,
        "end": 10.1525,
        "speaker": 0,
        "language": "zh"
      }
    ]
  }
]
```

### 2️⃣ 构建训练数据集

使用 `build_dataset.py` 将原始数据转换为训练格式：

```bash
python scripts/build_dataset.py \
  --data_dir <your_raw_data_dir> \
  --output_dir <your_output_dir> \
  --num_proc 8 \
  --test_samples 100 \
  --random_seed 42
```

**参数说明**：
- `--data_dir`: 原始数据目录（包含 Emilia 预处理后的 .json 和 .m4a 文件）
- `--output_dir`: 输出目录，会自动创建 `train/` 和 `test/` 子目录
- `--num_proc`: 并行处理进程数
- `--test_samples`: 测试集样本数量
- `--random_seed`: 随机种子

## 🔥 训练

### 💻 训练硬件

本项目在 **NVIDIA RTX 5090 (32GB)** 上训练。

### ⚡ 训练命令

**准备 Qwen3 模型**：

`model.llm_model_path` 可以是：
- **本地路径**：如 `./Qwen3-0.6B`（需提前下载）
- **Hugging Face 模型名**：如 `Qwen/Qwen3-0.6B`（会自动下载，但首次训练会较慢）

推荐提前下载到本地：
```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
```

**训练命令**：

```bash
python3 scripts/mel_train.py \
  dataset.train_dataset_path=<your_train_data_path> \
  dataset.valid_dataset_path=<your_valid_data_path> \
  model.llm_model_path=./Qwen3-0.6B \
  model.rfmel.batch_mul=2 \
  training.batch_size=4 \
  dataset.max_tokens=1024 \
  training.num_workers=16 \
  training.learning_rate=0.0001 \
  training.log_dir=<your_log_dir> \
  training.diffusion_extra_steps=4 \
  training.check_val_every_n_epoch=1 \
  model.use_skip_connection=true \
  model.estimator.hidden_dim=512 \
  model.estimator.intermediate_dim=1536 \
  model.estimator.num_layers=8
```

### 🚄 多卡训练

```bash
# 使用 2 张 GPU
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/mel_train.py \
  dataset.train_dataset_path=<your_train_data_path> \
  dataset.valid_dataset_path=<your_valid_data_path> \
  model.llm_model_path=Qwen3-0.6B \
  model.rfmel.batch_mul=2 \
  training.batch_size=8 \
  dataset.max_tokens=1024 \
  training.num_workers=16 \
  training.learning_rate=0.0001 \
  training.log_dir=<your_log_dir> \
  training.diffusion_extra_steps=4 \
  training.check_val_every_n_epoch=1 \
  model.use_skip_connection=true \
  model.estimator.hidden_dim=512 \
  model.estimator.intermediate_dim=1536 \
  model.estimator.num_layers=8
```

**注意**：
- Lightning 会自动检测并使用所有可用 GPU，使用 DDP 策略
- 根据您的硬件配置，可能需要调整 `batch_size`、`batch_mul`、`max_tokens` 等参数

## 📤 导出模型

训练完成后，导出模型用于推理：

```bash
python scripts/mel_export_checkpoint.py \
  --ckpt_path <your_checkpoint_path>/last.ckpt \
  --output_path ./exported_model/
```

或者直接指定 checkpoints 目录（自动选择最新的）：
```bash
python scripts/mel_export_checkpoint.py \
  --ckpt_path <your_checkpoint_dir>/ \
  --output_path ./exported_model/
```

导出后会生成：
- `model.ckpt`: 模型权重
- `model.yaml`: 推理配置

## 🎤 推理

```bash
python3 scripts/mel_inference.py \
  --model_path <your_model_dir>/ \
  --text example_data/transcript/fanren_short.txt \
  --ref_audio fanren08 \
  --output_path output/generated \
  --dtype bfloat16
```

**输出文件**：
- `output/generated.wav`: 生成的音频
- `output/generated.png`: Mel 频谱图
- `output/generated.npy`: Mel 频谱数组

### 🎧 参考音频说明

`--ref_audio` 参数指定参考音频的名称（不含扩展名），脚本会从 `example_data/voice_prompts/` 目录读取对应的 `.wav` 和 `.txt` 文件：

```
example_data/voice_prompts/
├── fanren08.wav          # 参考音频
├── fanren08.txt          # 参考音频对应的文本
├── fanren09.wav
└── fanren09.txt
```

可以添加自己的参考音频，只需将音频文件和对应的文本文件放入该目录即可。

### ⚙️ 参数说明

- `--model_path`: 导出的模型目录或 .ckpt 文件路径
- `--text`: 要合成的文本，或文本文件路径
- `--ref_audio`: 参考音频名称（不含扩展名），可用逗号分隔多个
- `--output_path`: 输出文件路径前缀（会生成 .wav, .png, .npy 三个文件）
- `--dtype`: 数据类型（float32/float16/bfloat16，默认 bfloat16）
- `--device`: 设备（cuda/cpu/mps，默认 cuda）
- `--temperature`: 采样温度（默认 0.7）
- `--top_p`: Top-p 采样（默认 0.7）
- `--max_new_tokens`: 最大生成 token 数（默认 1024）
- `--chunk_method`: 文本分块方法（speaker/word/none，默认 speaker）
- `--seed`: 随机种子（默认 42）

## 📁 项目结构

```
ar-dit-mel/
├── ar/                      # 自回归模型
│   ├── armel.py            # ARMel 主模型
│   ├── qwen.py             # Qwen3 LLM
│   └── mel_generate.py     # Mel 生成
├── rfwave/                  # 扩散模型
│   ├── mel_model.py        # RFMel 模型
│   ├── mel_processor.py    # Mel 处理器
│   └── estimator.py        # 扩散 Estimator
├── dataset/                 # 数据集
├── scripts/                 # 训练和推理脚本
│   ├── build_dataset.py    # 构建数据集
│   ├── mel_train.py        # 训练脚本
│   ├── mel_export_checkpoint.py  # 导出模型
│   └── mel_inference.py    # 推理脚本
└── configs/                 # 配置文件
```

## 📜 许可证

MIT License

## 📚 相关论文

- **Autoregressive Diffusion Transformer for Text-to-Speech Synthesis**
  Zhijun Liu, et al.
  [arXiv:2406.05551](https://arxiv.org/abs/2406.05551)

- **VibeVoice Technical Report**
  Zhiliang Peng, et al.
  [arXiv:2508.19205](https://arxiv.org/abs/2508.19205)

- **VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning**
  Yixuan Zhou, et al.
  [arXiv:2509.24650](https://arxiv.org/abs/2509.24650)

## 🙏 致谢

本项目基于以下开源项目：
- [Qwen3](https://github.com/QwenLM/Qwen) - 语言模型 🤖
- [Amphion](https://github.com/open-mmlab/Amphion) - 数据预处理 🎵
- [Vocos](https://github.com/gemelo-ai/vocos) - Vocoder 🔊
- [RFWave](https://github.com/bfs18/rfwave) - Diffusion Backbone 🌊
- [VoxCPM](https://github.com/OpenBMB/VoxCPM) - 架构参考 💡
- [Higgs-Audio](https://github.com/boson-ai/higgs-audio) - 数据模板 📋

