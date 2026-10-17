"""量化 DiT：GGUF 加载 / 转换。复用 Forge 自带的 gguf 包，不新增依赖。

- 只替换 DiT；Qwen3.5 文本编码器、VAE、tokenizer、scheduler 仍来自官方 Base / SFT 目录。
- 量化权重以 uint8 buffer 形式常驻（随 offload 正常搬运），forward 时按块反量化为 bf16 计算。
- 加载支持 Forge gguf 包能反量化的全部类型（Q8_0、Q5_K、Q4_K、Q6_K、Q3_K、Q2_K…）；
  转换器只能生成 Python 实现了量化的类型（Q8_0 / Q5_1 / Q4_1 / Q5_0 / Q4_0 / BF16）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from modules_forge.packages import gguf
from backend.operations_gguf import QUANTS_MAPPING

from .pipeline import MOSSImageTransformer

ARCH = "moss_image"
CONVERT_TYPES = {
    "Q8_0：约 5.3G，画质几乎不变（推荐）": gguf.GGMLQuantizationType.Q8_0,
    "Q5_1：约 3.5G，画质略降": gguf.GGMLQuantizationType.Q5_1,
    "Q4_1：约 3.0G，画质有所下降": gguf.GGMLQuantizationType.Q4_1,
}
# 只量化 42 个 block 中的大矩阵（与社区 ConvRot 量化范围一致），其余保持 bf16
_QUANT_KEY = re.compile(r"^layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)\.weight$")


class GGUFLinear(nn.Module):
    """无 bias 的量化线性层；forward 时反量化为输入 dtype。"""

    is_gguf_linear = True
    compute_dtype = torch.bfloat16  # 与官方 bf16 DiT 计算精度一致

    def __init__(self, raw, qtype, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.qtype = qtype
        self.gguf_cls = QUANTS_MAPPING[qtype]
        # 借用 Forge 的 bake：把原始块重排成便于 GPU 反量化的布局（scale 用 fp16 保存）
        holder = SimpleNamespace(data=raw, baked=False, computation_dtype=torch.float16)
        self.gguf_cls.bake(holder)
        # 用不求导的 Parameter 而不是 buffer：「逐层搬运」只会把参数移出显卡，buffer 会整体常驻显卡
        self.blocks = nn.Parameter(holder.data, requires_grad=False)

    def dequantized(self, dtype):
        block_size, type_size = gguf.GGML_QUANT_SIZES[self.qtype]
        holder = SimpleNamespace(baked=True, computation_dtype=torch.float16)
        weight = self.gguf_cls.dequantize_blocks_pytorch(self.blocks, block_size, type_size, holder)
        return weight.reshape(self.out_features, self.in_features).to(dtype)

    def forward(self, x):
        return F.linear(x, self.dequantized(x.dtype))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, {self.qtype.name}"


_HADAMARD = {}


def _hadamard(size, device):
    """comfy-kitchen 的规则 Hadamard：4x4 对称基的 Kronecker 幂，归一化后对称且正交。"""
    key = (size, str(device))
    if key not in _HADAMARD:
        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
        h = h4
        while h.shape[0] < size:
            h = torch.kron(h, h4)
        if h.shape[0] != size:
            raise ValueError(f"ConvRot 分组大小必须是 4 的幂：{size}")
        _HADAMARD[key] = (h / size ** 0.5).to(device=device, dtype=torch.bfloat16)
    return _HADAMARD[key]


def _unpack_int4(packed):
    """低半字节为偶数列，按有符号 4 bit 解释（与 comfy-kitchen _unpack_int4_row_major 一致）。"""
    x = packed.to(torch.int16)
    lo, hi = x & 0x0F, (x >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], -1).reshape(*packed.shape[:-1], -1)


class ConvRotLinear(nn.Module):
    """ComfyUI ConvRot 量化线性层（INT8 / W4A4），forward 时反量化回原始基：W = (q × scale) @ H。"""

    is_gguf_linear = True  # 供 LoRA 识别为可挂载的线性层
    compute_dtype = torch.bfloat16

    def __init__(self, qweight, scale, kind, group_size, in_features, out_features, bias=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.kind, self.group_size = kind, group_size
        # 同上：量化权重做成不求导的 Parameter，才能被「逐层搬运」移出显卡
        self.qweight = nn.Parameter(qweight, requires_grad=False)
        self.scale = nn.Parameter(scale.reshape(-1, 1).float(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.to(torch.bfloat16), requires_grad=False)

    def dequantized(self, dtype):
        q = _unpack_int4(self.qweight) if self.kind == "int4" else self.qweight
        w_rot = q.to(torch.bfloat16) * self.scale.to(torch.bfloat16)
        h = _hadamard(self.group_size, w_rot.device)
        w = (w_rot.reshape(self.out_features, -1, self.group_size) @ h).reshape(self.out_features, self.in_features)
        return w.to(dtype)

    def forward(self, x):
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.dequantized(x.dtype), bias)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, convrot_{self.kind}"


def list_quant_files(root):
    """models/MOSS-Image 根目录下的单文件 DiT：*.gguf 与 *.safetensors（ConvRot / bf16）。"""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix in (".gguf", ".safetensors"))


def load_quant_transformer(path, config_path):
    path = Path(path)
    if path.suffix == ".gguf":
        return load_gguf_transformer(path, config_path)
    return load_safetensors_transformer(path, config_path)


def _strip_prefix(name):
    for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ComfyUI 原生格式（Comfy-Org/MOSS-Image 等）把每层的 q / k / v 合并成一个 proj.qkv（按行依次拼接），
# 已用 Comfy-Org bf16 文件逐位验证：qkv.chunk(3, 0) == (官方 q, k, v)。加载时按行拆回三层。
def _qkv_parts(module_name):
    """layers.N.proj.qkv -> [layers.N.proj.q, .k, .v]；不是合并层时返回 None"""
    if module_name.endswith(".proj.qkv"):
        base = module_name[: -len("qkv")]
        return [base + c for c in "qkv"]
    return None


# 加载时忽略的附加张量：量化参数、ComfyUI 的注意力配置
_SKIP_SUFFIXES = (".comfy_quant", ".weight_scale", ".input_scale", ".comfy_attention.config")


def load_safetensors_transformer(path, config_path):
    """ComfyUI 单文件 DiT：ConvRot INT8（每层 comfy_quant）/ W4A4（_quantization_metadata）/ 普通 bf16。"""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    with init_empty_weights():
        transformer = MOSSImageTransformer(**config)
    modules = dict(transformer.named_modules())
    params = dict(transformer.named_parameters())
    loaded, quantized = set(), 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        names = {_strip_prefix(k): k for k in f.keys()}
        quant_layers = {}
        for name, raw in names.items():
            if name.endswith(".comfy_quant"):
                text = bytes(f.get_tensor(raw).tolist()).decode("utf-8")
                quant_layers[name[: -len(".comfy_quant")]] = json.loads(text)
        quant_layers.update(json.loads(meta.get("_quantization_metadata", "{}")).get("layers", {}))
        for module_name, info in quant_layers.items():
            fmt = info.get("format", "")
            if fmt == "int8_tensorwise" and info.get("convrot"):
                kind = "int8"
            elif fmt == "convrot_w4a4":
                kind = "int4"
            else:
                raise ValueError(f"不支持的 ComfyUI 量化格式：{module_name} -> {info}")
            qweight = f.get_tensor(names[module_name + ".weight"])
            scale = f.get_tensor(names[module_name + ".weight_scale"]).reshape(-1)
            bias = f.get_tensor(names[module_name + ".bias"]) if module_name + ".bias" in names else None
            # 合并的 qkv 按行拆成三层（旋转只作用于输入维度，按输出行拆分不影响反量化）
            targets = _qkv_parts(module_name) or [module_name]
            n = len(targets)
            for index, target in enumerate(targets):
                old = modules.get(target)
                if not isinstance(old, nn.Linear):
                    raise ValueError(f"量化层 {module_name} 不是 MOSS-Image DiT 的线性层（这不是 MOSS-Image 的模型文件？）")
                rows = slice(index * old.out_features, (index + 1) * old.out_features)
                part_q = qweight[rows] if n > 1 else qweight
                part_s = scale[rows] if n > 1 else scale
                part_b = None if bias is None else (bias[rows] if n > 1 else bias)
                expect_in = old.in_features // 2 if kind == "int4" else old.in_features
                if tuple(part_q.shape) != (old.out_features, expect_in) or part_s.numel() != old.out_features:
                    raise ValueError(f"{module_name} 的量化张量形状不符：{tuple(qweight.shape)} / {tuple(scale.shape)}")
                parent_name, child = target.rsplit(".", 1)
                setattr(modules[parent_name], child, ConvRotLinear(
                    part_q.contiguous(), part_s.contiguous(), kind, int(info.get("convrot_groupsize", 256)),
                    old.in_features, old.out_features, part_b,
                ))
                loaded.update({target + ".weight", target + ".bias"})
                quantized += 1
            loaded.update({module_name + ".weight", module_name + ".bias"})
        for name, raw in names.items():
            if name in loaded or name.endswith(_SKIP_SUFFIXES) or name == "rope.bands":
                continue
            parts = _qkv_parts(name[: -len(".weight")]) if name.endswith(".weight") else None
            if parts is not None:  # 非量化的合并 qkv（Comfy-Org bf16）
                for target, value in zip(parts, f.get_tensor(raw).chunk(3, 0)):
                    set_module_tensor_to_device(transformer, target + ".weight", "cpu", value=value.to(torch.bfloat16), dtype=torch.bfloat16)
                    loaded.add(target + ".weight")
                continue
            if name not in params:
                raise ValueError(f"文件中存在 MOSS-Image DiT 没有的张量：{name}（这不是 MOSS-Image 的 DiT 文件）")
            set_module_tensor_to_device(transformer, name, "cpu", value=f.get_tensor(raw).to(torch.bfloat16), dtype=torch.bfloat16)
            loaded.add(name)
    missing = [name for name in params if name not in loaded]  # 替换前的官方参数表
    if missing:
        raise ValueError(f"DiT 文件缺少 {len(missing)} 个张量，例如 {missing[:3]}")
    transformer.rope.bands = transformer.rope._default_bands(torch.device("cpu"))
    print(f"[MOSS-Image] 已加载单文件 DiT：{path.name}，ConvRot 量化层 {quantized} 个。")
    return transformer.eval()


def load_gguf_transformer(path, config_path):
    """按官方 transformer/config.json 建空模型，再填入 GGUF 张量。"""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    with init_empty_weights():
        transformer = MOSSImageTransformer(**config)
    reader = gguf.GGUFReader(str(path))
    modules = dict(transformer.named_modules())
    params = dict(transformer.named_parameters())
    loaded, quantized = set(), 0
    for tensor in reader.tensors:
        name = str(tensor.name)
        shape = tuple(int(v) for v in reversed(tensor.shape))
        qtype = tensor.tensor_type
        name = _strip_prefix(name)
        if name == "rope.bands":
            continue  # RoPE 频率按配置重新计算
        if name.endswith(_SKIP_SUFFIXES):
            continue
        data = torch.from_numpy(np.array(tensor.data))
        # 合并的 qkv（ComfyUI 原生格式转出的 GGUF）：量化块按行存储，按行拆成三份即可
        parts = _qkv_parts(name[: -len(".weight")]) if name.endswith(".weight") else None
        targets = [p + ".weight" for p in parts] if parts else [name]
        chunks = data.chunk(3, 0) if parts else [data]
        for target, chunk in zip(targets, chunks):
            if target not in params:
                raise ValueError(f"GGUF 中存在 MOSS-Image DiT 没有的张量：{name}（这不是 MOSS-Image 的 GGUF）")
            part_shape = tuple(params[target].shape) if parts else shape
            if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
                value = chunk.contiguous().view(torch.int16).view(torch.bfloat16) if qtype == gguf.GGMLQuantizationType.BF16 else chunk
                # 与官方 bf16 加载一致：norm / embedding 等非量化张量统一为 bf16
                set_module_tensor_to_device(transformer, target, "cpu", value=value.reshape(part_shape).to(torch.bfloat16), dtype=torch.bfloat16)
            else:
                if qtype not in QUANTS_MAPPING:
                    raise ValueError(f"不支持的 GGUF 量化类型：{qtype.name}（{name}）")
                module_name = target.rsplit(".", 1)[0]
                parent_name, child = module_name.rsplit(".", 1)
                old = modules[module_name]
                if not isinstance(old, nn.Linear) or old.bias is not None or part_shape != tuple(old.weight.shape):
                    raise ValueError(f"{name} 不是可量化的无 bias 线性层或形状不符：{shape}")
                setattr(modules[parent_name], child, GGUFLinear(chunk.contiguous(), qtype, old.in_features, old.out_features))
                quantized += 1
            loaded.add(target)
    missing = [name for name in params if name not in loaded]
    if missing:
        raise ValueError(f"GGUF 缺少 {len(missing)} 个张量，例如 {missing[:3]}")
    # RoPE 频率是按配置重新计算的 fp32 buffer
    transformer.rope.bands = transformer.rope._default_bands(torch.device("cpu"))
    print(f"[MOSS-Image] 已加载 GGUF DiT：{Path(path).name}，量化层 {quantized} 个。")
    return transformer.eval()


def convert_to_gguf(bundle_dir, out_path, qtype):
    """生成器：官方 bf16 DiT 分片 -> 单文件 GGUF，逐张量 yield 进度。"""
    transformer_dir = Path(bundle_dir) / "transformer"
    index = json.loads(next(transformer_dir.glob("*.safetensors.index.json")).read_text(encoding="utf-8"))
    names = sorted(k for k in index["weight_map"] if k != "rope.bands")  # RoPE 频率加载时重算
    out_path = Path(out_path)
    partial = out_path.with_name(out_path.name + ".part")
    writer = gguf.GGUFWriter(str(partial), arch=ARCH)
    writer.add_string("moss_image.source", Path(bundle_dir).name)
    writer.add_string("moss_image.quantization", qtype.name)
    handles = {}
    try:
        for count, name in enumerate(names, 1):
            shard = index["weight_map"][name]
            if shard not in handles:
                handles[shard] = safe_open(str(transformer_dir / shard), framework="pt", device="cpu")
            tensor = handles[shard].get_tensor(name)
            # 非量化张量保存为 BF16（官方权重本就是 bf16，转换无损）
            target = qtype if _QUANT_KEY.match(name) else gguf.GGMLQuantizationType.BF16
            array = gguf.quants.quantize(tensor.float().numpy(), target)
            writer.add_tensor(name, array, raw_shape=array.shape, raw_dtype=target)
            if count % 20 == 0 or count == len(names):
                yield f"转换中 {count}/{len(names)}：{name}"
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    finally:
        handles.clear()
    partial.replace(out_path)
    yield f"完成：{out_path}（{out_path.stat().st_size / 1024**3:.2f} GiB）"


# ---------------------------------------------------------------- 量化文本编码器

def list_text_encoder_files(root):
    """models/MOSS-Image/text_encoders 下的单文件 Qwen3.5（ComfyUI ConvRot INT8 等）。"""
    folder = Path(root) / "text_encoders"
    return sorted(p.name for p in folder.glob("*.safetensors")) if folder.is_dir() else []


def _quant_layers(f, names):
    layers = {}
    for name, raw in names.items():
        if name.endswith(".comfy_quant"):
            layers[name[: -len(".comfy_quant")]] = json.loads(bytes(f.get_tensor(raw).tolist()).decode("utf-8"))
    meta = json.loads((f.metadata() or {}).get("_quantization_metadata", "{}")).get("layers", {})
    layers.update({_strip_prefix(k): v for k, v in meta.items()})
    return layers


def _dequant_rows(q, scale, group_size, chunk=8192):
    """INT8 ConvRot 嵌入表整体反量化为 bf16（分块，避免一次占用大量内存）。"""
    h = _hadamard(group_size, "cpu").float()
    out = torch.empty(q.shape, dtype=torch.bfloat16)
    for start in range(0, q.shape[0], chunk):
        rows = q[start:start + chunk].float() * scale[start:start + chunk].reshape(-1, 1).float()
        out[start:start + chunk] = (rows.reshape(rows.shape[0], -1, group_size) @ h).reshape(rows.shape).to(torch.bfloat16)
    return out


def load_qwen35_quant_file(path, config_root):
    """ComfyUI 单文件 Qwen3.5 文本编码器：INT8 ConvRot 线性层按需反量化，嵌入表加载时反量化为 bf16。
    只读取 language_model（视觉塔不参与 MOSS 文生图）；键名与官方 HF 分片一致，可带 model. 前缀。"""
    from .qwen35 import Qwen35Config, Qwen35TextEncoder

    config = Qwen35Config.from_pretrained(config_root)
    with torch.device("meta"):
        model = Qwen35TextEncoder(config)
    modules = dict(model.named_modules())
    expected = set(model.state_dict())
    loaded, quantized = set(), 0

    def ours(name):
        name = name.removeprefix("model.")
        return "model." + name.removeprefix("language_model.") if name.startswith("language_model.") else None

    with safe_open(str(path), framework="pt", device="cpu") as f:
        names = {k: k for k in f.keys()}
        for raw_name, info in _quant_layers(f, names).items():
            module_name = ours(raw_name)
            if module_name is None:
                continue  # 视觉塔
            fmt = info.get("format", "")
            if not (fmt == "int8_tensorwise" and info.get("convrot")):
                raise ValueError(
                    f"暂不支持该文本编码器量化格式：{fmt}（目前支持 ComfyUI INT8 ConvRot；W4A8 GPTQ 请改用 INT8 版或官方 bf16）"
                )
            prefix = raw_name if raw_name + ".weight" in names else "model." + raw_name
            q, scale = f.get_tensor(prefix + ".weight"), f.get_tensor(prefix + ".weight_scale")
            group = int(info.get("convrot_groupsize", 256))
            module = modules.get(module_name)
            parent_name, child = module_name.rsplit(".", 1)
            if isinstance(module, nn.Embedding):
                weight = _dequant_rows(q, scale, group)
                setattr(modules[parent_name], child, nn.Embedding.from_pretrained(weight, freeze=True))
            elif isinstance(module, nn.Linear) and module.bias is None and tuple(q.shape) == tuple(module.weight.shape):
                setattr(modules[parent_name], child, ConvRotLinear(q, scale, "int8", group, module.in_features, module.out_features))
                quantized += 1
            else:
                raise ValueError(f"量化层 {raw_name} 与 Qwen3.5 结构不符：{tuple(q.shape)}")
            loaded.add(module_name + ".weight")
        for raw_name in names:
            name = ours(raw_name)
            if name is None or name in loaded or raw_name.endswith((".comfy_quant", ".weight_scale")):
                continue
            if name not in expected:
                raise ValueError(f"文本编码器文件中存在未知张量：{raw_name}（这不是 MOSS-Image 的 Qwen3.5？）")
            set_module_tensor_to_device(model, name, "cpu", value=f.get_tensor(raw_name).to(torch.bfloat16), dtype=torch.bfloat16)
            loaded.add(name)
    missing = sorted(expected - loaded)
    if missing:
        raise ValueError(f"文本编码器缺少 {len(missing)} 个张量，例如 {missing[:3]}")
    model.eval().requires_grad_(False)
    print(f"[MOSS-Image] 已加载量化文本编码器：{Path(path).name}，ConvRot 线性层 {quantized} 个。")
    return model
