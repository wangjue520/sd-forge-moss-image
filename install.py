"""Forge 每次启动（加载界面之前）都会运行本文件：检查并给 Forge Neo 本体打上 MOSS-Image 需要的小补丁。
已经打过时什么都不做；Forge 升级覆盖了这些文件，下次启动会自动重新打上。补丁内容见 lib_moss_image/forge_patch.py。"""
import importlib.util
import os
from pathlib import Path

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("moss_forge_patch", here / "lib_moss_image" / "forge_patch.py")
forge_patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forge_patch)

# Forge 运行 install.py 时把 WebUI 根目录放在 PYTHONPATH 第一位；兜底用扩展目录的上两级
candidates = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p] + [str(here.parents[1])]
root = next((Path(p) for p in candidates if (Path(p) / "modules" / "processing.py").is_file()), None)
if root is None:
    print("[MOSS-Image] 没有找到 WebUI 根目录，跳过本体适配。")
else:
    try:
        changed, message = forge_patch.apply(root)
        if changed:
            print(f"[MOSS-Image] 已自动适配 Forge Neo：{', '.join(changed)}")
        elif "不是 Forge Neo" in message:
            print(f"[MOSS-Image] {message}")
    except Exception as error:  # 不影响 WebUI 正常启动
        print(f"[MOSS-Image] 自动适配 Forge Neo 失败（不影响其他功能）：{error}")
