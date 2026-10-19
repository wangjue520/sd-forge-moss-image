# Changelog

## 0.2.0

- 新增 **GUIDE.md**：安装、模型、Layout 编辑器、JSON 导入/编译、局部重绘、LoRA、量化、常见问题。
- **可视化 Layout 编辑器**：在 txt2img / img2img 面板里直接拖框、双击编辑物体 / 角色 / 文字，
  填写整体描述、meta、canvas，点「写入正向提示词」即可。
- **导入外部 JSON**：画布上方的「导入 JSON」可以粘贴草稿或最终格式，直接还原成可视化框。
- **草稿 JSON 编译**：支持官方 prompt-enhancer 的 `abs / xyxy` 草稿，自动编译成
  `abs_centralized / yxyx`；JSON 后面的 `<lora:...>` 会保留。
- **真正的局部重绘**：img2img 的「只裁剪蒙版区域」会把蒙版包围盒裁出来、放大到当前分辨率
  重绘，再贴回原图，蒙版外像素不变。
- **生成中预览**：采样过程中每隔 N 步解码一张缩略图，显示在 Forge 进度条区域；可关闭或调间隔。
- 修复 Hires 第二遍布局缩放、布局 JSON 与 LoRA 标签混排、空蒙版处理等细节。
