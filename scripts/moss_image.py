"""UI Preset 选 moss 时接管生成：生成经 PROCESS_HOOKS 交给 MOSS-Image。

模型沿用 Forge 习惯：顶部 Checkpoint 选 MOSS-Image-Base / SFT / 量化 DiT，「VAE / Text Encoder」
可选量化 Qwen3.5；下方 MOSS-Image 面板只放采样相关选项，并自动显示当前所用模型。

支持 txt2img、Hires. fix、img2img / 局部重绘、DiT LoRA（<lora:名称:权重>）、Layout JSON、
中断 / 跳过 / 进度条，以及一键下载官方权重。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.nn import functional as F

from lib_moss_image.download import CATALOG, MEMORY_HINT, SOURCES, download_items, gpu_info, item_choices, recommend
from lib_moss_image.layout import validate_layout
from lib_moss_image.pipeline import Interrupted
from lib_moss_image import checkpoints
from lib_moss_image.runtime import MODEL_NAMES, available_configs, available_models, manager, model_roots, models_root, parse_layers
from modules import extra_networks, images, infotext_utils, sd_models, scripts, shared
from modules.infotext_utils import PasteField
from modules.processing import Processed, StableDiffusionProcessingImg2Img, create_binary_mask, fix_seed
from modules.shared import opts, state
try:
    from modules_forge.presets import CHECKPOINT_PROVIDERS, PRESET_CHECKPOINTS, PROCESS_HOOKS
except ImportError:
    # 启动时跳过了 install.py（--skip-install）：现在补上本体适配，重启一次后生效
    from lib_moss_image import forge_patch
    from modules import paths

    forge_patch.apply(paths.script_path)
    raise ImportError("[MOSS-Image] 已自动适配 Forge Neo 本体，请重启 WebUI 一次后使用。") from None

PRESET = "moss"

SLG_LABELS = {
    "negative": "开启（官方默认）",
    "off": "关闭",
    "standard": "加强版（每步多算一遍，更慢）",
}
MEMORY_LABELS = {
    "model": "按需放进显卡（推荐，24G 显卡约用 15G）",
    "none": "全部常驻显卡（最快，需要约 30G 显存）",
    "sequential": "逐层搬运（最省显存，速度很慢）",
}
# API 未通过 alwayson_scripts 传参时使用的官方默认值
DEFAULT_OPTIONS = {
    "slg_mode": "negative",
    "layers": "22",
    "memory_mode": "model",
    "keep_loaded": False,
}


def is_moss_preset():
    return opts.forge_preset == PRESET


def align_16(value):
    value = int(value)
    if value <= 0:
        raise ValueError("宽高必须为正数。")
    return max(16, math.floor(value / 16 + 0.5) * 16)


# ---------------------------------------------------------------- LoRA

def find_lora(name):
    """复用 Forge 内置 LoRA 列表（Lora 标签页可点选插入）；找不到时直接扫描 LoRA 目录。"""
    try:
        import networks  # extensions-builtin/sd_forge_lora

        if not networks.available_network_aliases:
            networks.list_available_networks()
        found = networks.available_network_aliases.get(name) or networks.available_networks.get(name)
        if found is not None:
            return found.filename
    except Exception:
        pass
    folders = [shared.cmd_opts.lora_dir, *(getattr(shared.cmd_opts, "lora_dirs", None) or [])]
    for folder in folders:
        if folder and os.path.isdir(folder):
            for path in shared.walk_files(folder, allowed_extensions=[".safetensors"]):
                if os.path.splitext(os.path.basename(path))[0] == name:
                    return path
    raise FileNotFoundError(f"找不到 LoRA：{name}（请放到 models/Lora）")


def resolve_loras(network_data):
    loras = []
    for params in network_data.get("lora", []):
        name = params.items[0]
        weight = float(params.items[1]) if len(params.items) > 1 else 1.0
        loras.append((name, find_lora(name), weight))
    for network in network_data:
        if network != "lora":
            print(f"[MOSS-Image] 不支持的额外网络 <{network}:...>，已忽略。")
    return loras


def lora_text(loras):
    return ", ".join(f"{name}:{weight:g}" for name, _, weight in loras) or None


# ---------------------------------------------------------------- Layout

def rescale_layout(prompt, width, height, target_w, target_h):
    """Hires 第二遍画布变大：按比例缩放 abs_centralized 像素坐标。"""
    layout = json.loads(prompt)
    sy, sx = target_h / height, target_w / width
    for item in layout["items"]:
        y1, x1, y2, x2 = item["bbox"]
        item["bbox"] = [round(y1 * sy), round(x1 * sx), round(y2 * sy), round(x2 * sx)]
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------- 图像工具

def to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)[None]


def prepare_mask(p, width, height):
    """返回 (像素 mask [H,W] 0~1, latent mask [1,1,h,w])；无 mask 时返回 (None, None)。"""
    if p.image_mask is None:
        return None, None
    mask = create_binary_mask(p.image_mask, round=getattr(p, "mask_round", True))
    if p.inpainting_mask_invert:
        mask = ImageOps.invert(mask)
    blur = int(getattr(p, "mask_blur", 0) or 0)
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    mask = images.resize_image(p.resize_mode, mask, width, height)
    pixel = torch.from_numpy(np.asarray(mask.convert("L"), dtype=np.float32) / 255.0)
    latent = F.interpolate(pixel[None, None], size=(height // 16, width // 16), mode="area")
    return pixel, latent


def composite(original, generated, pixel_mask):
    a = np.asarray(original.convert("RGB"), dtype=np.float32)
    b = np.asarray(generated.convert("RGB"), dtype=np.float32)
    m = pixel_mask.numpy()[..., None]
    return Image.fromarray(np.clip(a * (1 - m) + b * m + 0.5, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- infotext

def make_infotext(p, index, prompt, settings, extra):
    parameters = {
        "Steps": int(p.steps),
        "Sampler": "Euler",
        "Schedule type": "Simple",
        "CFG scale": float(p.cfg_scale),
        "Seed": int(p.all_seeds[index]),
        "Size": f"{p.width}x{p.height}",
        "RNG": "CPU",
        # 键名用 "Shift"，PNG 信息读回时会填到原生 Shift 滑块
        "Shift": float(settings["shift"]),
        **extra,
        # 与普通模型一致：Model 指顶部 Checkpoint，Module 1 指「VAE / Text Encoder」里的量化文本编码器
        "Model": settings["checkpoint"],
        "Module 1": settings["te"].name if settings["te"] is not None else None,
        "Module 2": settings["vae"].name if settings["vae"] is not None else None,
        "MOSS model": settings["model"],
        "MOSS sampler": "Euler flow-matching",
        "MOSS SLG mode": settings["slg_mode"],
        "MOSS SLG layers": ",".join(str(x) for x in settings["layers"]),
        "MOSS SLG scale": 3.0 if settings["slg_mode"] == "standard" else 0.0,
        "MOSS SLG interval": "0.01-1.0" if settings["slg_mode"] == "standard" else "0-1",
        "MOSS memory": settings["memory_mode"],
        "MOSS keep loaded": bool(settings["keep_loaded"]),
        "MOSS LoRA": settings["lora_text"],
    }
    text = ", ".join(
        key if key == value else f"{key}: {infotext_utils.quote(value)}"
        for key, value in parameters.items() if value is not None
    )
    negative = p.all_negative_prompts[index]
    negative_text = f"\nNegative prompt: {negative}" if negative else ""
    return f"{prompt}{negative_text}\n{text}".strip()


# ---------------------------------------------------------------- 生成

def _slg_kwargs(slg_mode, layers):
    return dict(
        uncond_skip_layers=layers if slg_mode == "negative" else (),
        slg_layers=layers if slg_mode == "standard" else (),
        slg_scale=3.0 if slg_mode == "standard" else 0.0,
        slg_start=0.01 if slg_mode == "standard" else 0.0,
        slg_stop=1.0,
    )


def _step_callback(step, total):
    state.sampling_step = step
    state.sampling_steps = total
    if state.interrupted or state.stopping_generation or state.skipped:
        raise Interrupted()


@torch.inference_mode()
def process_moss(p):
    """PROCESS_HOOKS 入口：替代 process_images，返回标准 Processed。"""
    is_img2img = isinstance(p, StableDiffusionProcessingImg2Img)
    options = {**DEFAULT_OPTIONS, **getattr(p, "moss_options", {})}
    slg_mode, memory_mode = options["slg_mode"], options["memory_mode"]
    keep_loaded = bool(options["keep_loaded"])
    # 模型来自顶部 Checkpoint（API 可用 override_settings.sd_model_checkpoint 指定）
    info = checkpoints.selected_checkpoint(p)
    if info is None:
        raise FileNotFoundError(
            f"没有找到 MOSS-Image 模型（查找目录：{'、'.join(str(r) for r in model_roots())}）。"
            "请在 MOSS-Image 面板的「模型下载」里一键下载。"
        )
    model, dit = checkpoints.resolve(info)
    modules = checkpoints.selected_modules(p)
    te, vae = modules["te"], modules["vae"]
    if modules["ignored"]:
        print(f"[MOSS-Image] 顶部「VAE / Text Encoder」中不适用于 MOSS 的文件已忽略：{modules['ignored']}")
    if model not in available_configs():
        raise FileNotFoundError(
            f"{info.name} 需要官方 MOSS-Image 配置包（{model}），但没有找到。"
            "请在 MOSS-Image 面板的「模型下载」里下载（只要配置包的话约 20MB）。"
        )
    if slg_mode not in SLG_LABELS or memory_mode not in MEMORY_LABELS:
        raise ValueError("MOSS SLG 或显存模式无效。")
    layers = parse_layers(options["layers"])
    # Shift 复用原生 Shift 滑块（Forge 中与 Distilled CFG 共用 distilled_cfg_scale 字段）
    shift = float(p.distilled_cfg_scale)
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("Shift 必须是正数。")

    p.width, p.height = align_16(p.width), align_16(p.height)
    p.sampler_name = "Euler"
    p.scheduler = "Simple"
    fix_seed(p)
    p.setup_prompts()
    total = len(p.all_prompts)
    p.all_seeds = [int(p.seed) + index for index in range(total)]
    p.all_subseeds = [int(p.subseed) + index for index in range(total)]
    p.sd_model_name = info.name
    p.sd_model_hash = None
    p.sd_vae_name = "FLUX.2 small decoder"
    p.sd_vae_hash = None

    # <lora:名称:权重> 从 prompt 中取出，只作用于 DiT；infotext 保留原 prompt 方便复现
    prompts, network_data = extra_networks.parse_prompts(list(p.all_prompts))
    loras = resolve_loras(network_data)
    for prompt in prompts:
        validate_layout(prompt, p.width, p.height)

    hires = not is_img2img and bool(getattr(p, "enable_hr", False))
    hr_extra = {}
    if hires:
        p.calculate_target_resolution()
        hr_w, hr_h = align_16(p.hr_upscale_to_x), align_16(p.hr_upscale_to_y)
        hr_steps = int(p.hr_second_pass_steps or p.steps)
        hr_prompts, hr_network_data = extra_networks.parse_prompts(list(p.all_hr_prompts))
        hr_loras = resolve_loras(hr_network_data)
        hr_extra = {
            "Denoising strength": float(p.denoising_strength),
            "Hires upscale": getattr(p, "hr_scale", None) if not (p.hr_resize_x or p.hr_resize_y) else None,
            "Hires resize": f"{hr_w}x{hr_h}" if (p.hr_resize_x or p.hr_resize_y) else None,
            "Hires steps": int(p.hr_second_pass_steps) or None,
            "Hires upscaler": p.hr_upscaler,
            "Hires CFG Scale": float(p.hr_cfg),
            "Hires Shift": float(p.hr_distilled_cfg),
        }
    if is_img2img:
        if not p.init_images:
            raise ValueError("img2img 需要输入图像。")
        if p.image_mask is not None and p.inpaint_full_res:
            print("[MOSS-Image] 「仅重绘蒙版区域」暂不支持，按整图重绘并保留蒙版外区域。")
        hr_extra = {"Denoising strength": float(p.denoising_strength)}
        if p.image_mask is not None:
            hr_extra["Mask blur"] = int(p.mask_blur)

    settings = {
        "model": model, "shift": shift, "slg_mode": slg_mode, "layers": layers,
        "memory_mode": memory_mode, "keep_loaded": keep_loaded, "lora_text": lora_text(loras), "te": te, "vae": vae,
        "checkpoint": info.name,
    }
    common = dict(model=model, offload=memory_mode, dit=dit, te=te, vae=vae, num_images_per_prompt=1,
                  callback=_step_callback, **_slg_kwargs(slg_mode, layers))

    # 首次接管时通过 Neo 自带接口完整卸载 Forge 当前模型。
    sd_models.unload_model_weights()
    output_images, infotexts = [], []
    state.job_count = total
    try:
        for index in range(total):
            if state.interrupted or state.stopping_generation:
                break
            state.skipped = False
            state.job = f"MOSS-Image {index + 1}/{total}"
            state.textinfo = "MOSS-Image：编码提示词 / 加载模型"
            seed = p.all_seeds[index]
            try:
                if is_img2img:
                    init = p.init_images[index % len(p.init_images)].convert("RGB")
                    init = images.resize_image(p.resize_mode, init, p.width, p.height)
                    pixel_mask, latent_mask = prepare_mask(p, p.width, p.height)
                    clean = manager.encode_image(model, memory_mode, to_tensor(init), dit, te, vae)
                    image = manager.generate(
                        shift=shift, loras=loras, prompt=prompts[index],
                        negative_prompt=p.all_negative_prompts[index],
                        height=p.height, width=p.width, num_inference_steps=int(p.steps),
                        guidance_scale=float(p.cfg_scale),
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                        image_latents=clean, strength=float(p.denoising_strength), mask=latent_mask, **common,
                    )[0]
                    if pixel_mask is not None:
                        image = composite(init, image, pixel_mask)
                else:
                    image = manager.generate(
                        shift=shift, loras=loras, prompt=prompts[index],
                        negative_prompt=p.all_negative_prompts[index],
                        height=p.height, width=p.width, num_inference_steps=int(p.steps),
                        guidance_scale=float(p.cfg_scale),
                        generator=torch.Generator(device="cpu").manual_seed(seed), **common,
                    )[0]
                    if hires:
                        if opts.save_images_before_highres_fix and p.save_samples():
                            first = make_infotext(p, index, p.all_prompts[index], settings, hr_extra)
                            images.save_image(image, p.outpath_samples, "", seed, p.all_prompts[index],
                                              opts.samples_format, info=first, p=p, suffix="-before-highres-fix")
                        state.textinfo = "MOSS-Image：Hires. fix"
                        upscaler = p.hr_upscaler
                        if upscaler in shared.latent_upscale_modes or upscaler in (None, "None"):
                            # MOSS latent 为 128 通道 packed 格式，latent 放大改用 Lanczos 像素放大
                            upscaler = None
                        upscaled = images.resize_image(0, image, hr_w, hr_h, upscaler_name=upscaler)
                        hr_prompt = hr_prompts[index]
                        if validate_layout(hr_prompt, p.width, p.height) is not None:
                            hr_prompt = rescale_layout(hr_prompt, p.width, p.height, hr_w, hr_h)
                        clean = manager.encode_image(model, memory_mode, to_tensor(upscaled), dit, te, vae)
                        image = manager.generate(
                            shift=float(p.hr_distilled_cfg), loras=hr_loras, prompt=hr_prompt,
                            negative_prompt=p.all_hr_negative_prompts[index],
                            height=hr_h, width=hr_w, num_inference_steps=hr_steps,
                            guidance_scale=float(p.hr_cfg),
                            generator=torch.Generator(device="cpu").manual_seed(seed),
                            image_latents=clean, strength=float(p.denoising_strength), **common,
                        )[0]
            except Interrupted:
                if state.skipped and not (state.interrupted or state.stopping_generation):
                    state.skipped = False
                    state.nextjob()
                    continue
                break

            info = make_infotext(p, index, p.all_prompts[index], settings, hr_extra)
            if p.save_samples():
                images.save_image(
                    image, p.outpath_samples, "", seed, p.all_prompts[index],
                    opts.samples_format, info=info, p=p,
                )
            if opts.enable_pnginfo:
                image.info["parameters"] = info
            output_images.append(image)
            infotexts.append(info)
            state.nextjob()

        index_of_first_image = 0
        if output_images and (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and len(output_images) > 1:
            grid = images.image_grid(output_images, p.batch_size)
            grid_info = infotexts[0]
            if opts.grid_save:
                images.save_image(
                    grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0],
                    opts.grid_format, info=grid_info, short_filename=not opts.grid_extended_filename,
                    grid=True, p=p,
                )
            if opts.return_grid:
                if opts.enable_pnginfo:
                    grid.info["parameters"] = grid_info
                output_images.insert(0, grid)
                infotexts.insert(0, grid_info)
                index_of_first_image = 1
    finally:
        if not keep_loaded:
            manager.release()

    if not infotexts:
        infotexts = [make_infotext(p, 0, p.all_prompts[0], settings, hr_extra)]
    return Processed(
        p, output_images,
        seed=p.all_seeds[0], info=infotexts[0], subseed=p.all_subseeds[0],
        all_prompts=p.all_prompts, all_negative_prompts=p.all_negative_prompts,
        all_seeds=p.all_seeds, all_subseeds=p.all_subseeds,
        index_of_first_image=index_of_first_image, infotexts=infotexts,
    )


PROCESS_HOOKS[PRESET] = process_moss
CHECKPOINT_PROVIDERS.append(checkpoints.register_checkpoints)
PRESET_CHECKPOINTS[PRESET] = checkpoints.is_moss_checkpoint
# 扩展晚于首次 list_models 加载：先注册一次，并给 moss 预设一个默认 Checkpoint（优先 SFT）
checkpoints.register_checkpoints()
if not getattr(opts, f"forge_checkpoint_{PRESET}", None) and checkpoints.default_checkpoint() is not None:
    opts.set(f"forge_checkpoint_{PRESET}", checkpoints.default_checkpoint().name)


def current_model_text():
    if not is_moss_preset():
        return ""
    return checkpoints.describe(checkpoints.selected_checkpoint(), checkpoints.selected_modules())


# ---------------------------------------------------------------- 下载

def download_dirs():
    """保存位置：MOSS 目录 + Forge 的文本编码器 / VAE 目录（有共享模型库时就是库里的 text_encoders / vae）。"""
    from modules import paths

    te_dirs = getattr(shared.cmd_opts, "text_encoder_dirs", None) or [str(Path(paths.models_path) / "text_encoder")]
    vae_dirs = getattr(shared.cmd_opts, "vae_dirs", None) or [str(Path(paths.models_path) / "VAE")]
    return {"moss": models_root(), "te": Path(te_dirs[0]), "vae": Path(vae_dirs[0])}


def _installed(key):
    item, dirs = CATALOG[key], download_dirs()
    if item.model is not None:
        return item.model in (available_models() if item.weights else available_configs())
    return (dirs[item.dest] / Path(item.path).name).is_file()


def model_status():
    checkpoints.register_checkpoints()
    lines = [f"- {'✅' if _installed(key) else '⬜'} {item.label}" for key, item in CATALOG.items()]
    dirs = download_dirs()
    return (
        "**本地已有**\n\n" + "\n".join(lines)
        + f"\n\n保存位置：主模型 `{dirs['moss']}`；文本编码器 `{dirs['te']}`；VAE `{dirs['vae']}`"
    )


def recommendation_text():
    name, vram = gpu_info()
    if name is None:
        return "没有检测到 NVIDIA 显卡，MOSS-Image 需要 CUDA 显卡。"
    keys, memory_mode, reason = recommend(vram)
    items = "、".join(CATALOG[key].label.split("（")[0] for key in keys)
    size = sum(CATALOG[key].size_gb for key in keys)
    return (
        f"检测到显卡：**{name}（{vram:.0f}G 显存）**\n\n"
        f"推荐下载：**{items}**（共约 {size:.1f}G）——{reason}。\n\n"
        f"下载后：顶部 Checkpoint 选{MEMORY_HINT[tuple(keys)]}，"
        f"「显存使用方式」选「{MEMORY_LABELS[memory_mode]}」。"
    )


def run_download(keys, source, custom_endpoint):
    if not keys:
        yield "请先勾选要下载的内容。"
        return
    endpoint = (custom_endpoint or "").strip() or SOURCES[source]
    text = "准备下载…"
    try:
        for text in download_items(list(keys), endpoint, download_dirs()):
            yield text
        sd_models.list_models()  # 让顶部 Checkpoint 能选到新模型（点刷新按钮即可看到）
        yield f"{text}\n\n下载完成，点顶部 Checkpoint 旁的刷新按钮即可选择。\n\n{model_status()}"
    except Exception as error:
        yield f"{text}\n\n下载中断（再次点击会从断点继续，也可以换一个下载源）：{error}"


def run_recommended(source, custom_endpoint):
    _, vram = gpu_info()
    keys, _, _ = recommend(vram)
    yield from run_download([key for key in keys if not _installed(key)] or keys, source, custom_endpoint)


def run_convert(model, label):
    from lib_moss_image.quant import CONVERT_TYPES, convert_to_gguf

    qtype = CONVERT_TYPES[label]
    if model not in available_models():
        yield f"本地没有完整的 {model}，请先下载。"
        return
    out = models_root() / f"MOSS-Image-{model}-{qtype.name}.gguf"
    if out.exists():
        yield f"已存在：{out}"
        return
    try:
        yield f"开始转换 {model} -> {qtype.name}（CPU 运算，约数分钟，期间可正常出图）"
        for text in convert_to_gguf(models_root() / model, out, qtype):
            yield text
    except Exception as error:
        yield f"转换失败：{error}"


# ---------------------------------------------------------------- UI

class MOSSImageGuard(scripts.Script):
    """非 moss 预设生成前释放 MOSS，防止两套模型抢显存。"""

    create_group = False

    def title(self):
        return "MOSS-Image Memory Guard"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def before_process(self, p, *args):
        if manager.loaded and not is_moss_preset():
            manager.release()


class MOSSImagePanel(scripts.Script):
    """MOSS 参数面板（txt2img / img2img），只在 UI Preset 为 moss 时显示。"""

    def __init__(self):
        self.panel = None

    def title(self):
        return "MOSS-Image"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"
        with gr.Accordion("MOSS-Image", open=True, visible=is_moss_preset(), elem_id=f"{tab}_moss_image_panel") as self.panel:
            self.current = gr.Markdown(current_model_text())
            with gr.Row():
                memory_mode = gr.Dropdown(
                    choices=[(label, value) for value, label in MEMORY_LABELS.items()],
                    value="model", label="显存使用方式",
                    info="MOSS 由三部分组成：文本编码器（理解提示词）、绘图主模型、VAE（把结果变成图片）。"
                         "「按需放进显卡」= 用到哪部分才放进显卡、用完放回内存，24G 显卡选它；"
                         "显存小于 16G 选「逐层搬运」；显存特别大才选「全部常驻显卡」。",
                )
            with gr.Row():
                slg_mode = gr.Dropdown(
                    choices=[(label, value) for value, label in SLG_LABELS.items()],
                    value="negative", label="结构增强（SLG）",
                    info="官方推荐的出图技巧：让画面结构、手部等更稳定。一般保持「开启（官方默认）」即可。",
                )
                layers = gr.Textbox(
                    value="22", label="结构增强作用的层号（高级）",
                    info="官方默认 22，一般不用改；多个层用英文逗号分隔，范围 0–41。",
                )
            keep_loaded = gr.Checkbox(
                value=False, label="生成完不卸载模型",
                info="勾选后连续出图不用每次重新加载（省约 30 秒），但会一直占用内存和显存；切换到其他模型时会自动释放。",
            )
            gr.Markdown(
                "**使用说明**\n\n"
                "- 模型在顶部 **Checkpoint** 里选：`MOSS-Image-SFT`（推荐，出图更规整）/ `MOSS-Image-Base`（原始底模），"
                "或文件名带 Q8_0、int8、int4 的**压缩版**（更省显存，画质略降，int4 降得明显）。\n"
                "- 想用压缩版的文本编码器，在顶部「VAE / Text Encoder」里选；不选就用官方原版。\n"
                "- 步数 / CFG / Shift 用页面上原有的滑块（官方推荐 50 / 3 / 5），宽高会自动对齐到 16 的倍数。\n"
                "- LoRA：提示词里写 `<lora:文件名:权重>`。\n"
                "- 排版（Layout）：把官方格式的 JSON 直接粘贴到正向提示词，宽高要和 JSON 里的画布一致。"
            )
            with gr.Accordion("模型下载", open=False):
                gr.Markdown(recommendation_text())
                with gr.Row():
                    source = gr.Dropdown(choices=list(SOURCES), value=next(iter(SOURCES)), label="下载源",
                                         info="国内一般选 HF 镜像或魔搭；压缩版文件只在 HF，选魔搭时会自动改走 HF 镜像。")
                    custom_endpoint = gr.Textbox(label="自定义下载地址（可选，一般留空）")
                recommended = gr.Button("一键下载推荐方案", variant="primary")
                items = gr.CheckboxGroup(choices=item_choices(), value=[], label="或者自己勾选要下载的内容")
                download = gr.Button("下载勾选的内容")
                status = gr.Markdown(model_status())
                recommended.click(run_recommended, inputs=[source, custom_endpoint], outputs=[status])
                download.click(run_download, inputs=[items, source, custom_endpoint], outputs=[status])
            with gr.Accordion("制作压缩版模型（GGUF）", open=False):
                from lib_moss_image.quant import CONVERT_TYPES

                gr.Markdown(
                    f"把官方绘图主模型压缩成一个 GGUF 文件（约 1 分钟），保存到 `{models_root()}`，"
                    "做好后点顶部 Checkpoint 旁的刷新按钮就能选到。Q8_0 体积减半、画质几乎不变；Q5 / Q4 更小，画质略降。\n\n"
                    "网上下载的压缩版（GGUF，或 ComfyUI 用的 int8 / int4 文件）放到这个目录或 `diffusion_models` 文件夹即可自动识别；"
                    "压缩版文本编码器放到 `text_encoders` 文件夹，然后在顶部「VAE / Text Encoder」里选。"
                )
                with gr.Row():
                    convert_model = gr.Dropdown(choices=list(MODEL_NAMES), value="SFT", label="用哪个官方模型制作")
                    convert_type = gr.Dropdown(choices=list(CONVERT_TYPES), value=next(iter(CONVERT_TYPES)), label="压缩程度")
                convert_status = gr.Markdown()
                convert = gr.Button("开始制作")
                convert.click(run_convert, inputs=[convert_model, convert_type], outputs=[convert_status])

        self.infotext_fields = [
            PasteField(slg_mode, "MOSS SLG mode"),
            PasteField(layers, "MOSS SLG layers"),
            PasteField(memory_mode, "MOSS memory"),
            PasteField(keep_loaded, "MOSS keep loaded"),
        ]
        self.paste_field_names = [field.label for field in self.infotext_fields]
        return [slg_mode, layers, memory_mode, keep_loaded]

    def after_component(self, component, **kwargs):
        # 顶部 UI Preset / Checkpoint / VAE 下拉框在 txt2img / img2img 之后创建，这里挂上联动
        elem_id = kwargs.get("elem_id")
        if self.panel is None:
            return
        if elem_id == "forge_ui_preset":
            component.change(
                fn=lambda preset: (gr.update(visible=preset == PRESET), current_model_text()),
                inputs=[component], outputs=[self.panel, self.current],
                queue=False, show_progress=False,
            )
            # 每次打开 / 刷新页面也按当前预设显示：有的 Forge 版本 UI Preset 在页面加载时才读当前设置，
            # 不会触发 change，界面构建时（启动时预设不是 moss）面板会一直隐藏
            from gradio.context import Context

            if Context.root_block is not None:
                Context.root_block.load(
                    fn=lambda: (gr.update(visible=is_moss_preset()), current_model_text()),
                    outputs=[self.panel, self.current], queue=False, show_progress=False,
                )
        elif elem_id in ("setting_sd_model_checkpoint", "setting_sd_modules"):
            # 等 Forge 把新选择写入设置后再刷新显示
            component.change(fn=lambda *_: current_model_text(), inputs=[], outputs=[self.current],
                             queue=False, show_progress=False)

    def before_process(self, p, slg_mode="negative", layers="22", memory_mode="model", keep_loaded=False):
        p.moss_options = {
            "slg_mode": slg_mode,
            "layers": layers,
            "memory_mode": memory_mode,
            "keep_loaded": keep_loaded,
        }
