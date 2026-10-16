"""MOSS-Image 模型发现、加载、offload 与驻留管理。"""
from __future__ import annotations

import gc
import json
import threading
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modules import paths

from .lora import LoRAManager
from .pipeline import Interrupted, MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
from .qwen35 import load_qwen35_text_encoder
from .vae import load_moss_vae, load_moss_vae_file


def _soft_empty_cache():
    # 延迟导入：模型发现、参数检查等纯 CPU 操作不会初始化 CUDA。
    from backend import memory_management

    memory_management.soft_empty_cache()


MODEL_NAMES = ("Base", "SFT")
OFFICIAL_DIT = "官方 bf16"
COMPONENTS = ("transformer", "text_encoder", "tokenizer", "vae", "scheduler")


def _library_roots():
    """共享模型库：启动参数里 --ckpt-dirs / --lora-dirs 等目录的上一级（如 F:/mod/loras -> F:/mod）。"""
    from modules import shared

    roots = []
    for attr in ("ckpt_dirs", "lora_dirs", "vae_dirs", "text_encoder_dirs"):
        for folder in getattr(shared.cmd_opts, attr, None) or []:
            parent = Path(folder).resolve().parent
            if parent not in roots and parent != Path(paths.models_path).resolve():
                roots.append(parent)
    return roots


def model_roots():
    """查找顺序：共享库/MOSS-Image（已存在时），再到 WebUI 的 models/MOSS-Image。"""
    candidates = [root / "MOSS-Image" for root in _library_roots()]
    local = Path(paths.models_path).resolve() / "MOSS-Image"
    return [root for root in candidates if root.is_dir()] + [local]


def models_root():
    """下载 / 转换的保存位置：优先共享模型库（如 F:/mod/MOSS-Image），否则 WebUI 的 models/MOSS-Image。"""
    library = _library_roots()
    return (library[0] / "MOSS-Image") if library else model_roots()[-1]


def model_path(name):
    """优先返回含完整官方权重的目录；只有配置包时也返回（权重由量化单文件提供）。"""
    if name not in MODEL_NAMES:
        raise ValueError(f"未知 MOSS-Image 版本：{name}；可选值为 {MODEL_NAMES}。")
    for weights in (WEIGHT_COMPONENTS, ()):
        for root in model_roots():
            if is_model_bundle(root / name, weights):
                return root / name
    return models_root() / name


WEIGHT_COMPONENTS = ("transformer", "text_encoder", "vae")


def is_model_bundle(root, weights=WEIGHT_COMPONENTS):
    """与官方 Gradio 相同，拒绝不完整的本地模型目录。
    weights：要求带官方权重的组件；传 () 只检查「配置包」（配置、tokenizer、scheduler）。"""
    try:
        root = Path(root)
        if not (root / "pipeline_moss_image.py").is_file():
            return False
        index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
        if not all(name in index and (root / name).is_dir() for name in COMPONENTS):
            return False
        for name in WEIGHT_COMPONENTS:
            component = root / name
            if not (component / "config.json").is_file():
                return False
            if name not in weights:
                continue
            indices = list(component.glob("*.safetensors.index.json"))
            if indices:
                for file in indices:
                    shards = json.loads(file.read_text(encoding="utf-8"))["weight_map"].values()
                    if not shards or not all((component / shard).is_file() for shard in shards):
                        return False
            elif not any(component.glob("*.safetensors")):
                return False
        tokenizer = root / "tokenizer"
        return (
            (root / "scheduler" / "scheduler_config.json").is_file()
            and (tokenizer / "tokenizer_config.json").is_file()
            and (tokenizer / "chat_template.jinja").is_file()
            and (
                (tokenizer / "tokenizer.json").is_file()
                or ((tokenizer / "vocab.json").is_file() and (tokenizer / "merges.txt").is_file())
            )
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def available_models():
    """含完整官方权重的版本（可直接作为 MOSS-Image-Base / SFT 使用）。"""
    return [name for name in MODEL_NAMES if is_model_bundle(model_path(name))]


def available_configs():
    """至少有配置包的版本（量化单文件需要它提供配置、tokenizer、scheduler）。"""
    return [name for name in MODEL_NAMES if is_model_bundle(model_path(name), ())]


def dit_choices():
    """DiT 权重选项：官方 bf16 + 各 MOSS-Image 目录下的 GGUF / ConvRot 单文件。"""
    from .quant import list_quant_files

    names = []
    for root in model_roots():
        names += [name for name in list_quant_files(root) if name not in names]
    return [OFFICIAL_DIT, *names]


def dit_path(dit):
    if dit in (None, "", OFFICIAL_DIT):
        return None
    if Path(dit).is_absolute():
        if not Path(dit).is_file():
            raise FileNotFoundError(f"找不到量化 DiT 文件：{dit}")
        return Path(dit)
    for root in model_roots():
        path = (root / dit).resolve()
        if path.parent == root.resolve() and path.is_file():
            return path
    raise FileNotFoundError(f"找不到量化 DiT 文件：{dit}（查找目录：{', '.join(map(str, model_roots()))}）")


def text_encoder_choices():
    """文本编码器选项：官方 bf16（随模型版本）+ 各 MOSS-Image/text_encoders 下的量化单文件。"""
    from .quant import list_text_encoder_files

    names = []
    for root in model_roots():
        names += [name for name in list_text_encoder_files(root) if name not in names]
    return [OFFICIAL_DIT, *names]


def text_encoder_path(name):
    if name in (None, "", OFFICIAL_DIT):
        return None
    if Path(name).is_absolute():
        if not Path(name).is_file():
            raise FileNotFoundError(f"找不到量化文本编码器：{name}")
        return Path(name)
    for root in model_roots():
        path = (root / "text_encoders" / name).resolve()
        if path.parent == (root / "text_encoders").resolve() and path.is_file():
            return path
    raise FileNotFoundError(f"找不到量化文本编码器：{name}")


def parse_layers(value, depth=42):
    if isinstance(value, (list, tuple)):
        layers = tuple(value)
    else:
        text = str(value or "").strip()
        layers = () if not text else tuple(int(item.strip()) for item in text.split(","))
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"SLG 层号必须是 0 到 {depth - 1} 的整数。")
    if len(set(layers)) != len(layers):
        raise ValueError("SLG 层号不能重复。")
    return layers


def _load_pipeline(root, offload, dit_file=None, te_file=None, vae_file=None):
    root = Path(root).resolve()
    # 只要求「没有用单文件替换」的组件带官方权重
    needed = tuple(name for name, override in zip(WEIGHT_COMPONENTS, (dit_file, te_file, vae_file)) if override is None)
    if not is_model_bundle(root, needed):
        missing = "、".join({"transformer": "主模型", "text_encoder": "文本编码器", "vae": "VAE"}[n] for n in needed)
        raise FileNotFoundError(
            f"MOSS-Image 目录不完整：{root}\n"
            f"缺少官方{missing}权重（或配置文件）。请在 MOSS-Image 面板「模型下载」里下载，"
            "或在顶部选择对应的单文件模型。"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("MOSS-Image 第一阶段需要 CUDA 显卡。")

    # 先在 CPU 加载，再按官方 inference_runtime.py 配置设备与 offload。
    if dit_file is None:
        transformer = MOSSImageTransformer.from_pretrained(
            root / "transformer", torch_dtype=torch.bfloat16, local_files_only=True
        ).eval()
    else:
        # 量化 DiT 只替换 transformer；文本编码器、VAE 等仍来自官方目录
        from .quant import load_quant_transformer

        transformer = load_quant_transformer(dit_file, root / "transformer" / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    tokenizer.chat_template = (root / "tokenizer" / "chat_template.jinja").read_text(encoding="utf-8")
    if te_file is None:
        text_encoder = load_qwen35_text_encoder(root / "text_encoder")
    else:
        # 量化文本编码器：结构配置仍取官方 text_encoder/config.json
        from .quant import load_qwen35_quant_file

        text_encoder = load_qwen35_quant_file(te_file, root / "text_encoder")
    vae = load_moss_vae(root / "vae") if vae_file is None else load_moss_vae_file(vae_file, root / "vae")
    pipe = MOSSImagePipeline(transformer, MOSSImageScheduler(), text_encoder, tokenizer, vae)

    device = "cuda"
    if offload == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    elif offload == "none":
        pipe.to(device)
    else:
        raise ValueError(f"未知显存模式：{offload}")
    return pipe


class ModelManager:
    def __init__(self):
        self.pipe = None
        self.key = None
        self.lock = threading.RLock()
        self.prompt_cache = {}
        self.loras = LoRAManager()

    @property
    def loaded(self):
        return self.pipe is not None

    def release(self):
        """通过 Neo 的内存管理入口清缓存；不会在普通模型生成后残留 MOSS。"""
        with self.lock:
            self.loras.remove()
            if self.pipe is not None:
                self.pipe.remove_all_hooks()
            self.pipe = None
            self.key = None
            self.prompt_cache.clear()
            gc.collect()
            _soft_empty_cache()

    def _prompt_embeddings(self, model, offload, prompt, negative_prompt, guidance_scale, max_sequence_length):
        """缓存同 prompt 编码；缓存放 CPU，避免长期占用 3090 显存。"""
        do_cfg = guidance_scale != 1.0
        key = (model, prompt, negative_prompt if do_cfg else None, max_sequence_length)
        if key not in self.prompt_cache:
            positive, positive_mask = self.pipe.encode_prompt(prompt, max_sequence_length)
            if do_cfg:
                negative, negative_mask = self.pipe.encode_prompt(negative_prompt, max_sequence_length)
            else:
                negative = negative_mask = None
            if len(self.prompt_cache) >= 8:
                self.prompt_cache.pop(next(iter(self.prompt_cache)))
            self.prompt_cache[key] = tuple(
                None if value is None else value.detach().to("cpu", copy=True)
                for value in (positive, positive_mask, negative, negative_mask)
            )

            if offload == "model":
                # 关键：DiT 开始前显式重建组件 hooks，确保文本编码器已经回到 CPU。
                self.pipe.maybe_free_model_hooks()
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"[MOSS-Image] Prompt 编码完成，文本编码器已卸载到 CPU；当前 CUDA 占用 {allocated:.2f} GiB。")
            elif offload == "sequential":
                # 顺序 hooks 在每个子模块 forward 后回收显存，不能对 meta 参数直接 .to()。
                _soft_empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"[MOSS-Image] Prompt 编码完成，顺序 offload 已回收文本编码器显存；当前 CUDA 占用 {allocated:.2f} GiB。")
            if offload != "none":
                _soft_empty_cache()
        return self.prompt_cache[key]

    def _ensure(self, model, offload, dit=None, te=None, vae=None):
        dit_file, te_file = dit_path(dit), text_encoder_path(te)
        vae_file = Path(vae) if vae else None
        key = (str(model_path(model).resolve()), offload, str(dit_file), str(te_file), str(vae_file))
        if key != self.key:
            self.release()
            self.pipe = _load_pipeline(model_path(model), offload, dit_file, te_file, vae_file)
            self.key = key

    def encode_image(self, model, offload, image, dit=None, te=None, vae=None):
        """img2img / Hires.fix：VAE 编码输入图（fp32）。"""
        with self.lock:
            try:
                self._ensure(model, offload, dit, te, vae)
                return self.pipe.encode_image(image)
            except Exception:
                self.release()
                raise

    def generate(self, model, offload, shift, loras=(), dit=None, te=None, vae=None, **kwargs):
        """loras: [(名称, 文件路径, 权重)]，只挂在 DiT 上；文本编码缓存不受影响。
        dit / te: None 或「官方 bf16」，或 MOSS-Image 目录下的量化单文件名。"""
        with self.lock:
            try:
                self._ensure(model, offload, dit, te, vae)
                torch.cuda.reset_peak_memory_stats()
                report = self.loras.apply(self.pipe.transformer, list(loras))
                for name, matched, skipped_te, unmatched in report:
                    print(f"[MOSS-Image] LoRA {name}: 匹配 {matched} 个 DiT 层，忽略文本编码器键 {skipped_te} 个，未匹配 {unmatched} 个。")
                self.pipe.scheduler = type(self.pipe.scheduler).from_config(
                    self.pipe.scheduler.config, shift=float(shift)
                )

                prompt = kwargs.pop("prompt")
                negative_prompt = kwargs.pop("negative_prompt", "")
                max_sequence_length = int(kwargs.pop("max_sequence_length", 4096))
                guidance_scale = float(kwargs["guidance_scale"])
                positive, positive_mask, negative, negative_mask = self._prompt_embeddings(
                    model, offload, prompt, negative_prompt, guidance_scale, max_sequence_length
                )
                kwargs.update(prompt_embeds=positive, prompt_attention_mask=positive_mask)
                if guidance_scale != 1.0:
                    kwargs.update(
                        negative_prompt_embeds=negative,
                        negative_prompt_attention_mask=negative_mask,
                    )
                images = self.pipe(**kwargs).images
                peak = torch.cuda.max_memory_reserved() / 1024**3
                print(f"[MOSS-Image] 本张峰值显存 {peak:.2f} GiB（显存使用方式：{offload}）")
                return images
            except Interrupted:
                # 中断 / 跳过不卸载模型，只把组件挪回 CPU
                self.pipe.maybe_free_model_hooks()
                _soft_empty_cache()
                raise
            except Exception:
                # OOM 或加载/编码失败都不能把半套组件或文本编码器留在显卡上。
                self.release()
                raise


manager = ModelManager()
