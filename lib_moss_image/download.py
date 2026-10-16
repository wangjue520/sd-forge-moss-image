"""MOSS-Image 下载：官方完整版 / 配置包 / 压缩版主模型 / 文本编码器 / VAE，并按显存推荐组合。

- 官方仓库支持 Hugging Face、HF 镜像、ModelScope；社区压缩版只在 Hugging Face（及其镜像）上。
- 不使用 huggingface_hub.snapshot_download：hf-mirror.com 重定向后缺少 X-Repo-Commit，新版会失败。
  这里直接走文件列表 API；大文件 16 路并发分块下载，分块各自断点续传。
"""
from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests

SOURCES = {
    "HF 镜像 (hf-mirror.com)": "https://hf-mirror.com",
    "Hugging Face": "https://huggingface.co",
    "ModelScope 魔搭": "https://www.modelscope.cn",
}
HF_REPOS = {"Base": "OpenMOSS-Team/MOSS-Image-Base", "SFT": "OpenMOSS-Team/MOSS-Image-SFT"}
MS_REPOS = {"Base": "openmoss/MOSS-Image-Base", "SFT": "openmoss/MOSS-Image-SFT"}
# 推理只需要这些；README、assets 展示图等不下载
NEEDED_DIRS = ("transformer/", "text_encoder/", "tokenizer/", "vae/", "scheduler/")
NEEDED_FILES = ("model_index.json", "pipeline_moss_image.py")
WEIGHT_DIRS = ("transformer/", "text_encoder/", "vae/")
PART = 32 * 1024 * 1024  # 并发分块大小
WORKERS = 16


@dataclass(frozen=True)
class Item:
    label: str
    dest: str  # "moss" / "te" / "vae"
    size_gb: float
    model: str | None = None  # 官方仓库：Base / SFT
    weights: bool = True  # 官方仓库：False = 只下载配置包
    repo: str | None = None  # 社区单文件：HF 仓库
    path: str | None = None


CATALOG = {
    "sft_full": Item("官方 SFT 完整版（主模型 + 文本编码器 + VAE，出图更规整）", "moss", 25.4, model="SFT"),
    "base_full": Item("官方 Base 完整版（原始底模，适合训练 LoRA）", "moss", 25.4, model="Base"),
    "sft_config": Item("官方配置包（用压缩版必需，只有配置和分词器）", "moss", 0.02, model="SFT", weights=False),
    "dit_int8": Item("压缩版主模型 INT8（画质接近原版）", "moss", 5.41,
                     repo="nif0/MOSS-Image-SFT-ComfyUI", path="diffusion_models/moss_image_sft_int8_convrot.safetensors"),
    "dit_int4": Item("压缩版主模型 INT4（最小，画质明显下降）", "moss", 3.04,
                     repo="toxicdog/MOSS-Image-SFT-INT4-ConvRot-ComfyUI", path="diffusion_models/moss_image_sft_int4_convrot.safetensors"),
    "te_int8": Item("压缩版文本编码器 INT8（显存约省一半）", "te", 8.45,
                    repo="Abiray/MOSS-Image-SFT-INT8-convrot", path="text_encoders/moss-image-sft-qwen35-int8-convrot.safetensors"),
    "vae_bf16": Item("VAE 单文件（bf16）", "vae", 0.125,
                     repo="nif0/MOSS-Image-SFT-ComfyUI", path="vae/moss_image_vae_bf16.safetensors"),
}


def item_choices():
    return [(f"{item.label} — 约 {item.size_gb:g}G" if item.size_gb >= 0.1 else f"{item.label} — 约 20M", key)
            for key, item in CATALOG.items()]


def _is_modelscope(endpoint):
    return "modelscope" in endpoint


def _needed(path, weights):
    if not (path in NEEDED_FILES or path.startswith(NEEDED_DIRS)):
        return False
    return weights or not (path.startswith(WEIGHT_DIRS) and path.endswith(".safetensors"))


def list_files(model, endpoint, session=None, weights=True):
    """官方仓库文件列表：[(相对路径, 字节数, 下载 URL)]。"""
    session = session or requests.Session()
    endpoint = endpoint.rstrip("/")
    if _is_modelscope(endpoint):
        repo = MS_REPOS[model]
        response = session.get(
            f"{endpoint}/api/v1/models/{repo}/repo/files",
            params={"Recursive": "true", "Revision": "master"}, timeout=60,
        )
        response.raise_for_status()
        files = [
            (f["Path"], int(f["Size"]),
             f"{endpoint}/api/v1/models/{repo}/repo?Revision=master&FilePath={quote(f['Path'])}")
            for f in response.json()["Data"]["Files"] if f["Type"] == "blob"
        ]
    else:
        repo = HF_REPOS[model]
        response = session.get(f"{endpoint}/api/models/{repo}/tree/main", params={"recursive": "true"}, timeout=60)
        response.raise_for_status()
        files = [
            (f["path"], int(f["size"]), f"{endpoint}/{repo}/resolve/main/{quote(f['path'])}")
            for f in response.json() if f["type"] == "file"
        ]
    files = [item for item in files if _needed(item[0], weights)]
    if not any(path.startswith("transformer/") for path, _, _ in files):
        raise RuntimeError(f"{endpoint} 返回的文件列表不完整，请换一个下载源。")
    return files


def _single_stream(session, url, partial, size):
    """不支持分块时的单连接续传。"""
    done = partial.stat().st_size if partial.is_file() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with session.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as response:
        if done and response.status_code != 206:
            done = 0
        response.raise_for_status()
        with open(partial, "ab" if done else "wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                yield done


def _download_file(session, url, target, size):
    """生成器：yield 已下载字节数。大文件 16 路并发分块，每块独立续传；大小一致视为已完成。"""
    target = Path(target)
    if target.is_file() and target.stat().st_size == size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if size <= 2 * PART:
        yield from _single_stream(session, url, partial, size)
    else:
        parts_dir = target.with_name(target.name + ".parts")
        parts_dir.mkdir(exist_ok=True)
        ranges = [(i, start, min(start + PART, size) - 1) for i, start in enumerate(range(0, size, PART))]
        progress = {i: (parts_dir / f"{i:05d}").stat().st_size if (parts_dir / f"{i:05d}").is_file() else 0
                    for i, _, _ in ranges}
        lock, errors = threading.Lock(), []

        def fetch(item):
            index, start, end = item
            path = parts_dir / f"{index:05d}"
            for attempt in range(8):
                have = path.stat().st_size if path.is_file() else 0
                if have == end - start + 1:
                    return
                try:
                    with requests.get(url, headers={"Range": f"bytes={start + have}-{end}", **session.headers},
                                      stream=True, timeout=60, allow_redirects=True) as response:
                        if response.status_code != 206:
                            raise IOError("服务器不支持分块下载")
                        with open(path, "ab") as handle:
                            for chunk in response.iter_content(1 << 20):
                                handle.write(chunk)
                                with lock:
                                    progress[index] += len(chunk)
                except Exception as error:  # 网络波动：稍后重试该分块
                    if attempt == 7:
                        errors.append(error)
                        return
                    time.sleep(2 + attempt)

        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(fetch, item) for item in ranges]
            while not all(f.done() for f in futures):
                time.sleep(1)
                yield sum(progress.values())
        if errors:
            raise IOError(f"{target.name} 部分分块下载失败（{errors[0]}），可再次点击继续。")
        with open(partial, "wb") as out:
            for index, _, _ in ranges:
                with open(parts_dir / f"{index:05d}", "rb") as handle:
                    shutil.copyfileobj(handle, out, 16 << 20)
        shutil.rmtree(parts_dir)
        yield size
    if partial.stat().st_size != size:
        raise IOError(f"{target.name} 大小不符：{partial.stat().st_size} / {size}，可再次点击继续下载。")
    partial.replace(target)


def _plan(key, endpoint, dirs, session):
    """一个下载项 -> [(保存路径, 字节数, URL)]。"""
    item = CATALOG[key]
    if item.model is not None:
        root = Path(dirs["moss"]) / item.model
        return [(root / path, size, url) for path, size, url in list_files(item.model, endpoint, session, item.weights)]
    if _is_modelscope(endpoint):
        endpoint = SOURCES["HF 镜像 (hf-mirror.com)"]  # 社区文件只在 HF，魔搭改走 HF 镜像
    url = f"{endpoint.rstrip('/')}/{item.repo}/resolve/main/{quote(item.path)}"
    head = session.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    size = int(head.headers.get("Content-Length") or head.headers.get("X-Linked-Size") or 0)
    return [(Path(dirs[item.dest]) / Path(item.path).name, size, url)]


def download_items(keys, endpoint, dirs):
    """生成器：依次下载所选项，边下载边 yield 进度文本（约每秒一次）。dirs: {"moss","te","vae"} 保存目录。"""
    session = requests.Session()
    session.headers["User-Agent"] = "sd-forge-moss-image"
    files = []
    for key in keys:
        files += _plan(key, endpoint, dirs, session)
    total = sum(size for _, size, _ in files)
    finished, started, last = 0, time.time(), 0.0
    for index, (target, size, url) in enumerate(files, 1):
        for attempt in range(3):
            try:
                for done in _download_file(session, url, target, size):
                    now = time.time()
                    if now - last >= 1.0:
                        last = now
                        current = finished + done
                        speed = current / max(now - started, 1e-6) / 1024**2
                        yield (f"[{index}/{len(files)}] {target.name}：{done / 1024**3:.2f}/{size / 1024**3:.2f} GiB，"
                               f"总计 {current / 1024**3:.2f}/{total / 1024**3:.2f} GiB，{speed:.1f} MiB/s")
                break
            except (requests.RequestException, IOError):
                if attempt == 2:
                    raise
                time.sleep(3)
        finished += size
    yield f"完成：共 {len(files)} 个文件，{total / 1024**3:.2f} GiB。"


def download_model(model, endpoint, root, only=None):
    """兼容旧接口：下载官方完整版到 root/<版本>（only 为测试用的路径过滤函数）。"""
    session = requests.Session()
    files = list_files(model, endpoint, session)
    if only is not None:
        files = [item for item in files if only(item[0])]
    for index, (path, size, url) in enumerate(files, 1):
        for _ in _download_file(session, url, Path(root) / model / path, size):
            pass
        yield f"[{index}/{len(files)}] {path}"
    yield f"完成：{model} 共 {len(files)} 个文件"


# ---------------------------------------------------------------- 按显存推荐

def gpu_info():
    import torch

    if not torch.cuda.is_available():
        return None, 0.0
    props = torch.cuda.get_device_properties(0)
    return props.name, props.total_memory / 1024**3


# 显存门槛来自 RTX 3090 实测峰值（1024x1024，见 README「显存与速度」）：
# 官方完整版 + 按需放进显卡 14.85 GiB；压缩版组合 + 按需 8.7 GiB；压缩版组合 + 逐层搬运 3.04 GiB
RECOMMENDATIONS = [
    (18, ["sft_full"], "model", "官方完整版，画质最好（实测峰值约 15G）"),
    (10, ["sft_config", "dit_int8", "te_int8", "vae_bf16"], "model", "压缩版主模型 + 压缩版文本编码器，画质接近原版（实测峰值约 8.7G）"),
    (0, ["sft_config", "dit_int8", "te_int8", "vae_bf16"], "sequential", "压缩版组合 +「逐层搬运」（实测峰值约 3G，速度较慢；模型放在内存里，内存建议 32G 以上）"),
]


MEMORY_HINT = {
    ("sft_full",): "「MOSS-Image-SFT」",
    ("sft_config", "dit_int8", "te_int8", "vae_bf16"):
        "「moss_image_sft_int8_convrot.safetensors」，「VAE / Text Encoder」选 INT8 文本编码器和 VAE",
}


def recommend(vram_gb):
    for threshold, keys, memory_mode, reason in RECOMMENDATIONS:
        if vram_gb >= threshold:
            return keys, memory_mode, reason
    return RECOMMENDATIONS[-1][1:]
