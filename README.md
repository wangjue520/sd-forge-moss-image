# SD Forge MOSS-Image

在 SD WebUI Forge Neo 里使用 OpenMOSS 的 **MOSS-Image**（Base / SFT），用法和其他模型一样：
顶部 **UI Preset 选 `moss`**，**Checkpoint 选 MOSS 模型**，然后正常出图。

扩展只使用 Forge 已有的依赖（不升级 torch / transformers / diffusers），Qwen3.5 文本编码器以纯 PyTorch 实现。

## 安装

**方式一（推荐）**：用 [WWY 启动器](https://github.com/wangjue520/forge-webui-launcher)的「常用插件」页勾选 MOSS-Image 安装。

**方式二**：WebUI 的 Extensions → Install from URL，填入本仓库地址：

```text
https://github.com/wangjue520/sd-forge-moss-image.git
```

或者手动 `git clone` 到 `extensions/sd-forge-moss-image`。

装好后**重启 WebUI**：

1. 首次启动会自动给 Forge Neo 本体打一个小补丁（新增 `moss` 预设等，见文末「本体改动」），不需要手动操作。
   如果启动时带了 `--skip-install`，补丁会在加载时补上，并提示再重启一次。
2. 顶部 UI Preset 选 `moss`，打开下方「MOSS-Image」面板的 **模型下载**，点「一键下载推荐方案」
   （会按你的显存推荐，也可以自己勾选）。下载完点 Checkpoint 旁的刷新按钮。

> 仅支持 **Forge Neo**（含 Neo · H3）；Classic 版没有 UI Preset，不支持。需要 NVIDIA 显卡。

### 模型放在哪里

扩展会根据启动参数里的模型目录（如 `--ckpt-dirs F:/mod/diffusion_models`）自动找到共享模型库，
没有共享库时使用 WebUI 自己的 `models`。以共享库 `F:\mod` 为例：

```text
F:\mod\
├── MOSS-Image\
│   ├── SFT\                      官方完整版或配置包（OpenMOSS-Team/MOSS-Image-SFT）
│   ├── Base\                     官方完整版（OpenMOSS-Team/MOSS-Image-Base）
│   ├── *.gguf                    压缩版主模型（GGUF，可在面板里自己制作）
│   └── *_convrot.safetensors     压缩版主模型（ComfyUI 社区 INT8 / INT4）
├── diffusion_models\             压缩版主模型放这里也能识别（和 ComfyUI 共用）
├── text_encoders\                压缩版文本编码器（Qwen3.5 INT8）
└── vae\                          MOSS 的 VAE 单文件（FLUX.2 VAE）
```

## 怎么用

| 位置 | 作用 |
|---|---|
| 顶部 **UI Preset** | 选 `moss`。步数 / CFG / Shift 自动变成官方推荐的 50 / 3 / 5，采样器 Euler |
| 顶部 **Checkpoint** | 只列 MOSS 模型：`MOSS-Image-SFT`（推荐）、`MOSS-Image-Base`、压缩版（Q8_0 / int8 / int4） |
| 顶部 **VAE / Text Encoder** | 不选 = 用官方原版；可选压缩版文本编码器、MOSS VAE 单文件。选了不适用的（SDXL VAE、T5 等）会在面板提示并忽略 |
| 原生参数 | 提示词、负向提示词（官方推荐留空）、步数、CFG、Shift、宽高（自动对齐 16）、seed、批次 |
| MOSS-Image 面板 | 显存使用方式、结构增强（SLG）、生成完不卸载模型、模型下载、制作压缩版 |

### 面板选项

- **显存使用方式**
  - 按需放进显卡（推荐）：文本编码器、主模型、VAE 用到哪个才放进显卡，用完放回内存。
  - 全部常驻显卡：最快，但官方完整版需要约 30G 显存。
  - 逐层搬运：显存最省，速度很慢，给小显存显卡用。
- **结构增强（SLG）**：官方推荐的技巧（负向分支跳过第 22 层），让结构、手部更稳定，保持默认即可。
- **生成完不卸载模型**：连续出图时省去每次约 30 秒的加载；切到其他模型时自动释放。

### 其他功能

- **LoRA**：提示词里写 `<lora:文件名:权重>`（LoRA 放在 LoRA 目录）。只作用于主模型，不修改文本编码器；
  支持 diffusers / PEFT、ComfyUI、kohya 三种键名格式，也能用在压缩版主模型上。
- **Hires. fix**：用原生的 Hires. fix。Latent 放大方式会自动改为 Lanczos 像素放大
  （MOSS 的 latent 不是普通 4 通道格式）。
- **img2img / 局部重绘**：原生 img2img 标签页。官方没有提供编辑模型，这里采用「VAE 编码 → 按重绘幅度加噪 → 重绘」；
  局部重绘在每步把蒙版外区域替换回原图，结束后再按蒙版合成。
- **Layout 排版**：把官方格式 JSON（`bbox_coord_space: abs_centralized`、`bbox_format: yxyx`）直接粘贴到正向提示词，
  原样传给模型。宽高必须和 JSON 的画布一致，超出画布、格式不对会给出具体错误提示；Hires 第二遍会按比例缩放坐标。
- **中断 / 跳过 / 进度条**：原生按钮即可，进度条显示采样步数。
- **PNG 图片信息**：完整记录模型、压缩版文件、SLG、LoRA 等参数，可读回并发送到 txt2img。

### 压缩版（量化）

| 类型 | 来源 | 说明 |
|---|---|---|
| GGUF Q8_0 / Q5_1 / Q4_1 | 面板「制作压缩版模型」一键生成（约 1 分钟） | 也能加载其他工具生成的 Q6_K / Q5_K / Q4_K 等 |
| ConvRot INT8 / INT4 主模型 | ComfyUI 社区（nif0、toxicdog 等） | 面板可直接下载 |
| ConvRot INT8 文本编码器 | ComfyUI 社区（Abiray 等） | 面板可直接下载 |

压缩版只替换主模型 / 文本编码器 / VAE 本身，配置和分词器仍来自官方目录：只用压缩版时下载「官方配置包」（约 20M）即可。

## 显存与速度（RTX 3090 24G，1024×1024）

| 主模型 | 文本编码器 | 显存使用方式 | 峰值显存 | 8 步耗时（含加载） | 画质（与官方版比较） |
|---|---|---|---:|---:|---|
| 官方 SFT | 官方 | 按需放进显卡 | 14.85 GiB | 61–75 秒 | 基准 |
| 官方 SFT | 官方 | 逐层搬运 | **3.04 GiB** | 107 秒 | 与上一行一致 |
| INT8 ConvRot | 官方 | 按需放进显卡 | 14.85 GiB | 66 秒 | 接近（PSNR 24.8 dB） |
| GGUF Q8_0 | 官方 | 按需放进显卡 | 14.85 GiB | 70 秒 | 接近（PSNR 26.5 dB） |
| INT8 ConvRot | INT8 | 按需放进显卡 | **8.7 GiB** | 49–55 秒 | 接近（PSNR 26.6 dB） |
| INT8 ConvRot | INT8 | 逐层搬运 | **3.04 GiB** | 80 秒 | 接近（PSNR 26.6 dB） |
| GGUF Q8_0 | INT8 | 逐层搬运 | **3.04 GiB** | 87 秒 | 最接近（PSNR 28.1 dB） |
| INT4 ConvRot | INT8 | 按需放进显卡 | 8.7 GiB | 46 秒 | 明显变差（PSNR 14.9 dB），不推荐 |

- 官方完整版 1024×1024、50 步一张约 150 秒（「按需放进显卡」）。
- 峰值出现在**文本编码器**编码提示词时，所以只压缩主模型不省显存；把文本编码器换成 INT8 才能从约 15G 降到约 9G。
- 画质对比：512×512、20 步、同一 seed 与官方版逐像素比较。PSNR 25 dB 以上时构图、内容、文字一致，只有细节差异。
- 面板「模型下载」按这张表推荐：显存 ≥ 18G 用官方完整版；10–18G 用压缩版组合；10G 以下用压缩版组合 +「逐层搬运」
  （模型放在内存里，内存建议 32G 以上）。

## 精度

- 主模型、文本编码器：bf16；VAE、初始噪声、CFG 与 Euler 更新：fp32（与官方一致）。
- 每张图使用独立的 CPU 随机数生成器，同 seed 同参数可逐像素复现。
- 与官方隔离环境（diffusers 0.38 / transformers 5.12）同参数对比：构图、内容、文字一致，细节因 bf16 GPU 计算顺序差异略有不同。

## 已知限制

- img2img「仅重绘蒙版区域」暂不支持（按整图重绘并保留蒙版外内容）。
- 生成过程中没有中间预览图，只有进度条。
- 社区 W4A8 GPTQ 文本编码器暂不支持（INT8 版可用）。
- INT4 主模型画质明显下降（每层权重误差约 15–18%，属于该文件本身）。
- 需要 NVIDIA CUDA 显卡。

## 本体改动

为了让 MOSS 像普通模型一样出现在 UI Preset 和 Checkpoint 中，`install.py` 会自动对 Forge Neo 做以下小改动
（只增加代码、不改原有逻辑；已改过就跳过；Forge 升级覆盖后下次启动自动重新打上；代码见 `lib_moss_image/forge_patch.py`）：

- `modules_forge/presets.py`：新增 `moss` 预设（Euler / Simple / 50 步 / CFG 3 / Shift 5）；新增扩展接口
  `PROCESS_HOOKS`（预设接管生成）、`CHECKPOINT_PROVIDERS`（扩展注册 Checkpoint）、`PRESET_CHECKPOINTS`（按预设过滤 Checkpoint 下拉框）。
- `modules/processing.py`：`process_images` 开头检查 `PROCESS_HOOKS`。
- `modules/sd_models.py`：`list_models` 结束时调用 `CHECKPOINT_PROVIDERS`。
- `modules_forge/main_entry.py`：Checkpoint 下拉框按当前预设过滤。

官方行为以 [MOSS-Image inference_runtime.py](https://github.com/OpenMOSS/MOSS-Image/blob/main/inference_runtime.py)
及 HF `pipeline_moss_image.py` 为准；模型与上游代码采用 Apache 2.0。ConvRot 反量化公式与 comfy-kitchen 一致。
