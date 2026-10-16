"""Qwen3.5 纯文本编码器。

结构移植自官方 ComfyUI 插件对应实现，并以 inference_runtime.py 实际调用的
HF Qwen3_5Model 数值行为为准校正。只保留 MOSS 推理需要的整段文本 prefill，
不包含视觉塔、文本生成和 KV cache，因此不需要 transformers>=5.3。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open


def _layer_types(count):
    return ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(count)]


@dataclass
class Qwen35Config:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    conv_kernel_size: int = 4
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    mrope_section: list = field(default_factory=lambda: [11, 11, 10])
    layer_types: list = field(default_factory=lambda: _layer_types(32))
    rms_norm_add: bool = True
    mlp_activation: str = "silu"
    qkv_bias: bool = False

    @classmethod
    def from_pretrained(cls, root):
        root = Path(root)
        raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if raw.get("model_type") != "qwen3_5":
            raise ValueError("MOSS-Image 需要 Qwen3.5 文本编码器。")
        text = raw["text_config"]
        rope = text["rope_parameters"]
        config = cls(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            partial_rotary_factor=text["partial_rotary_factor"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            conv_kernel_size=text["linear_conv_kernel_dim"],
            max_position_embeddings=text["max_position_embeddings"],
            rms_norm_eps=text["rms_norm_eps"],
            rope_theta=rope["rope_theta"],
            mrope_section=list(rope["mrope_section"]),
            layer_types=list(text["layer_types"]),
            mlp_activation=text["hidden_act"],
            qkv_bias=text["attention_bias"],
        )
        if len(config.layer_types) != config.num_hidden_layers:
            raise ValueError("Qwen3.5 layer_types 与层数不一致。")
        return config


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, add=True, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.add = add
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x):
        # HF Qwen3.5 明确在 fp32 中完成归一化和权重乘法，再转回输入精度。
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        weight = 1.0 + self.weight.float() if self.add else self.weight.float()
        return (output * weight).type_as(x)


class RMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x, gate):
        # Gated RMSNorm 的 weight 不加 1，顺序与 HF Qwen3_5RMSNormGated 一致。
        input_dtype = x.dtype
        hidden = x.float()
        hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + self.eps)
        hidden = self.weight * hidden.to(input_dtype)
        hidden = hidden * F.silu(gate.float())
        return hidden.to(input_dtype)


class MLP(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        if config.mlp_activation == "silu":
            self.activation = F.silu
        elif config.mlp_activation == "gelu_pytorch_tanh":
            self.activation = lambda value: F.gelu(value, approximate="tanh")
        else:
            raise ValueError(f"不支持的 Qwen3.5 MLP 激活：{config.mlp_activation}")

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64):
    """与 HF 回退实现相同，DeltaNet 核心在 fp32 中计算。"""
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (query.shape[-1] ** -0.5)

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        recurrent_state = (
            recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:2], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(config.hidden_size, conv_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False, device=device, dtype=dtype)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_value_heads, bias=False, device=device, dtype=dtype)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_value_heads, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.dt_bias = nn.Parameter(torch.empty(self.num_value_heads, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.empty(self.num_value_heads, device=device, dtype=dtype))
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, kernel_size=self.conv_kernel_size, groups=conv_dim,
            padding=self.conv_kernel_size - 1, bias=False, device=device, dtype=dtype,
        )
        self.norm = RMSNormGated(self.value_head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x)
        beta = self.in_proj_b(x).sigmoid()
        a = self.in_proj_a(x)
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len]).transpose(1, 2)
        query, key, value = mixed_qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        query = query.reshape(batch_size, seq_len, -1, self.key_head_dim)
        key = key.reshape(batch_size, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch_size, seq_len, -1, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            repeat = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        core_attn_out = torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.value_head_dim), z.reshape(-1, self.value_head_dim))
        return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


def precompute_partial_rope(head_dim, rotary_dim, position_ids, theta, device, mrope_section, dtype):
    """HF Qwen3.5 的 interleaved MRoPE；纯文本也使用相同的频率排列。"""
    del head_dim
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    theta_numerator = torch.arange(0, rotary_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (theta_numerator / rotary_dim))
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        index = slice(offset, length, 3)
        freqs_t[..., index] = freqs[dim, ..., index]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    cos = emb.cos().to(dtype=dtype).unsqueeze(1)
    sin = emb.sin().to(dtype=dtype).unsqueeze(1)
    return cos, sin


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_partial_rope(xq, xk, freqs_cis, rotary_dim):
    cos, sin = freqs_cis
    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]
    xq_rot = xq_rot * cos + rotate_half(xq_rot) * sin
    xk_rot = xk_rot * cos + rotate_half(xk_rot) * sin
    return (
        torch.cat((xq_rot, xq_pass), dim=-1),
        torch.cat((xk_rot, xk_pass), dim=-1),
    )


class GatedAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        self.q_proj = nn.Linear(config.hidden_size, self.inner_size * 2, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(self, x, attention_mask, freqs_cis):
        batch_size, seq_len, _ = x.shape
        qg = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim * 2)
        xq, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
        gate = gate.reshape(batch_size, seq_len, -1)
        xk = self.k_proj(x)
        xv = self.v_proj(x)
        xq = self.q_norm(xq).transpose(1, 2)
        xk = self.k_norm(xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xq, xk = apply_partial_rope(xq, xk, freqs_cis, self.rotary_dim)
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            xk = xk.repeat_interleave(repeat, dim=1)
            xv = xv.repeat_interleave(repeat, dim=1)
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=attention_mask, is_causal=attention_mask is None
        )
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(output * gate.sigmoid())


class Qwen35TransformerBlock(nn.Module):
    def __init__(self, config, index, device=None, dtype=None):
        super().__init__()
        self.layer_type = config.layer_types[index]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, device=device, dtype=dtype)
        elif self.layer_type == "full_attention":
            self.self_attn = GatedAttention(config, device=device, dtype=dtype)
        else:
            raise ValueError(f"未知 Qwen3.5 层类型：{self.layer_type}")
        self.mlp = MLP(config, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(self, x, attention_mask, freqs_cis):
        if self.layer_type == "linear_attention":
            hidden = self.linear_attn(self.input_layernorm(x))
        else:
            hidden = self.self_attn(self.input_layernorm(x), attention_mask, freqs_cis)
        x = x + hidden
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen35Transformer(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            Qwen35TransformerBlock(config, index, device=device, dtype=dtype)
            for index in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        rotary_dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        freqs_cis = precompute_partial_rope(
            self.config.head_dim, rotary_dim, position_ids,
            self.config.rope_theta, x.device, self.config.mrope_section, x.dtype,
        )

        # 无 padding 时让 SDPA 使用自己的 causal kernel，与 HF 的 fast path 一致。
        mask = None
        if attention_mask is not None and not torch.all(attention_mask):
            mask = 1.0 - attention_mask.to(x.dtype).reshape(
                attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
            ).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)
            if seq_len > 1:
                causal_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device)
                causal_mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)
                mask = mask + causal_mask

        for layer in self.layers:
            x = layer(x, mask, freqs_cis)
        return self.norm(x)


class Qwen35TextEncoder(nn.Module):
    """向官方 MOSS pipeline 提供 transformers 风格的返回接口。"""

    def __init__(self, config, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.model = Qwen35Transformer(config, device=device, dtype=dtype)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, **kwargs):
        del use_cache, kwargs
        hidden = self.model(input_ids, attention_mask=attention_mask)
        if return_dict:
            return SimpleNamespace(last_hidden_state=hidden)
        return (hidden,)


def load_qwen35_text_encoder(root):
    """从 HF 分片只读取 language_model，避免把未使用的视觉塔载入内存。"""
    root = Path(root)
    config = Qwen35Config.from_pretrained(root)
    with torch.device("meta"):
        model = Qwen35TextEncoder(config)
    expected = set(model.state_dict())

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"缺少文本编码器分片索引：{index_path}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    language_map = {
        key: "model." + key.removeprefix("language_model.")
        for key in weight_map if key.startswith("language_model.")
    }
    incoming = set(language_map.values())
    if incoming != expected:
        missing = sorted(expected - incoming)
        unexpected = sorted(incoming - expected)
        raise ValueError(f"Qwen3.5 权重键不匹配；缺失={missing[:8]}，未知={unexpected[:8]}")

    shards = {}
    for source, mapped in language_map.items():
        shards.setdefault(weight_map[source], []).append((source, mapped))
    for filename, keys in sorted(shards.items()):
        path = (root / filename).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"文本编码器分片缺失或路径非法：{path}")
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            state = {mapped: shard.get_tensor(source) for source, mapped in keys}
        model.load_state_dict(state, strict=False, assign=True)

    meta = [name for name, parameter in model.named_parameters() if parameter.device.type == "meta"]
    if meta:
        raise ValueError(f"Qwen3.5 仍有未加载参数：{meta[:8]}")
    # 官方要求文本编码器权重为 bf16；视觉塔不参与 MOSS 文生图。
    model.to(dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    return model
