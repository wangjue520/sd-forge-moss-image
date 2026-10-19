// MOSS-Image visual Layout editor for Forge Neo.
//
// Draws boxes over the current txt2img / img2img canvas, turns them into the
// official MOSS layout JSON, and writes it into the positive prompt. The
// editor is intentionally dependency-free: it only uses DOM APIs and the
// global helpers Forge already provides (gradioApp / onUiLoaded).
(function () {
  "use strict";

  const COLORS = { object: "#e0b34a", character: "#5cc8ff", text: "#ff7a7a" };
  const TYPE_NAMES = { object: "物体", character: "角色", text: "文字" };
  const MIN_BOX = 6;

  // Forge normally loads extension style.css automatically. Keep a fallback so
  // the panel is still tidy if a build only loads javascript/.
  const scriptUrl = document.currentScript && document.currentScript.src;
  if (scriptUrl) {
    const cssUrl = new URL("../style.css", scriptUrl).href;
    const exists = Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
      .some((link) => link.href === cssUrl || link.getAttribute("href") === "../style.css");
    if (!exists) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = cssUrl;
      link.dataset.mossLayoutStyle = "1";
      document.head.append(link);
    }
  }

  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (key === "class") node.className = value;
      else if (key === "style") node.style.cssText = value;
      else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2), value);
      } else if (value !== undefined && value !== null && value !== false) {
        node.setAttribute(key, value === true ? "" : String(value));
      }
    }
    for (const child of children.flat ? children.flat() : children) {
      if (child === null || child === undefined) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function appRoot() {
    return typeof gradioApp === "function" ? gradioApp() : document;
  }

  function isDarkTheme() {
    const root = document.documentElement;
    const body = document.body;
    if ((root && root.classList.contains("dark")) || (body && body.classList.contains("dark"))) return true;
    if (!body) return false;
    const candidates = [body, document.querySelector(".gradio-container"), root];
    for (const node of candidates) {
      if (!node) continue;
      let color = "";
      try { color = getComputedStyle(node).backgroundColor || ""; } catch (error) { continue; }
      const parts = color.match(/\d+(?:\.\d+)?/g);
      if (!parts || parts.length < 3) continue;
      if (parts.length >= 4 && Number(parts[3]) === 0) continue;
      const [r, g, b] = parts.slice(0, 3).map(Number);
      return (r * 0.299 + g * 0.587 + b * 0.114) < 128;
    }
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function queryTextarea(elemId) {
    const root = appRoot();
    const host = root.querySelector(`#${elemId}`) || document.querySelector(`#${elemId}`);
    return host ? host.querySelector("textarea, input[type='text']") : null;
  }

  function setTextarea(elemId, value) {
    const box = queryTextarea(elemId);
    if (!box) return false;
    box.value = value;
    box.dispatchEvent(new Event("input", { bubbles: true }));
    box.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  function getTextarea(elemId) {
    const box = queryTextarea(elemId);
    return box ? box.value : "";
  }

  function findJson(text) {
    const source = String(text || "");
    const start = source.indexOf("{");
    if (start < 0) return null;
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let i = start; i < source.length; i += 1) {
      const ch = source[i];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) return source.slice(start, i + 1);
      }
    }
    return null;
  }

  function tagsAfterJson(text) {
    const source = String(text || "");
    const json = findJson(source);
    const tail = json ? source.slice(source.indexOf(json) + json.length) : source;
    return (tail.match(/<(?:lora|lyco|hypernet):[^>]+>/gi) || []).join(" ");
  }

  function readNumber(input, fallback) {
    const value = Number(input && input.value);
    return Number.isFinite(value) && value > 0 ? Math.round(value) : fallback;
  }

  function clampBox(box, w, h) {
    box.x1 = Math.max(0, Math.min(w, Number(box.x1) || 0));
    box.y1 = Math.max(0, Math.min(h, Number(box.y1) || 0));
    box.x2 = Math.max(0, Math.min(w, Number(box.x2) || 0));
    box.y2 = Math.max(0, Math.min(h, Number(box.y2) || 0));
    if (box.x2 < box.x1) [box.x1, box.x2] = [box.x2, box.x1];
    if (box.y2 < box.y1) [box.y1, box.y2] = [box.y2, box.y1];
    return box;
  }

  function build(host, tab) {
    const state = {
      boxes: [],
      selected: -1,
      editing: -1,
      summary: "",
      meta: "",
      canvas: "single",
      w: 1024,
      h: 1024,
      drag: null,
      hasBackground: false,
      lastBoxClick: null,
      lastRowClick: null,
      importing: false,
    };

    host.innerHTML = "";
    host.classList.toggle("mle-dark", isDarkTheme());
    const toolbar = el("div", { class: "mle-toolbar" });
    const toolbarLeft = el("div", { class: "mle-toolbar-left" });
    const toolbarRight = el("div", { class: "mle-toolbar-right" });
    const sizeLabel = el("span", { class: "mle-size-pill", title: "跟随当前 Width / Height" }, "1024 × 1024");
    const fileLabel = el("label", { class: "mle-upload", title: "只影响画布背景，不会改变生成结果" }, "背景图");
    const file = el("input", { type: "file", accept: "image/*", class: "mle-file" });
    fileLabel.append(file);
    const importButton = el("button", { type: "button", class: "mle-btn quiet" }, "导入 JSON");
    const readButton = el("button", { type: "button", class: "mle-btn quiet" }, "读取提示词");
    const draftButton = el("button", { type: "button", class: "mle-btn quiet" }, "导出草稿");
    const compileButton = el("button", { type: "button", class: "mle-btn quiet" }, "编译 JSON");
    const writeButton = el("button", { type: "button", class: "mle-btn primary" }, "写入正向提示词");
    const clearButton = el("button", { type: "button", class: "mle-btn quiet danger" }, "清空");
    importButton.title = "粘贴一段 Layout JSON，直接还原成画布上的框";
    readButton.title = "从当前正向提示词里读取已有布局";
    draftButton.title = "把当前框导出为 abs / xyxy 草稿";
    compileButton.title = "编译成 abs_centralized / yxyx 最终格式";
    writeButton.title = "编译并替换正向提示词里的布局，保留 LoRA 标签";
    toolbarLeft.append(sizeLabel, fileLabel, importButton);
    toolbarRight.append(readButton, draftButton, compileButton, clearButton, writeButton);
    toolbar.append(toolbarLeft, toolbarRight);

    const frame = el("div", { class: "mle-frame" });
    const bg = el("img", { class: "mle-bg", alt: "" });
    const boxes = el("div", { class: "mle-boxes" });
    const hint = el("div", { class: "mle-hint" },
      el("span", { class: "mle-hint-title" }, "拖拽画框"),
      el("span", { class: "mle-hint-sub" }, "双击框编辑类型、描述和文字"),
    );
    frame.append(bg, boxes, hint);

    const summary = el("textarea", {
      class: "mle-summary", rows: "3",
      placeholder: "描述整个画面：场景、光线、风格……（会成为 summary.caption）",
    });
    const meta = el("input", {
      type: "text", class: "mle-meta",
      placeholder: "poster, product photo…",
    });
    const canvas = el("select", { class: "mle-canvas" },
      el("option", { value: "single" }, "单张画面"),
      el("option", { value: "multi_panel" }, "多格分镜"),
      el("option", { value: "collage" }, "拼贴"),
    );
    const addButton = el("button", { type: "button", class: "mle-btn small" }, "+ 添加");
    const countBadge = el("span", { class: "mle-count" }, "0");
    const list = el("div", { class: "mle-list" });
    const readable = el("pre", { class: "mle-readable" });
    const status = el("div", { class: "mle-status" }, "画框后点「写入正向提示词」。草稿格式会在生成时自动编译。");
    const side = el("div", { class: "mle-side" },
      el("section", { class: "mle-card" },
        el("div", { class: "mle-card-title" }, "整体描述"),
        summary,
      ),
      el("section", { class: "mle-card" },
        el("div", { class: "mle-card-title" }, "画面信息"),
        el("div", { class: "mle-field-row" },
          el("label", { class: "mle-field" }, el("span", { class: "mle-label" }, "Meta"), meta),
          el("label", { class: "mle-field mle-field-canvas" }, el("span", { class: "mle-label" }, "Canvas"), canvas),
        ),
      ),
      el("section", { class: "mle-card mle-elements-card" },
        el("div", { class: "mle-card-head" },
          el("span", { class: "mle-card-title" }, "布局元素"),
          countBadge,
          addButton,
        ),
        list,
      ),
      el("section", { class: "mle-card mle-summary-card" },
        el("details", {},
          el("summary", {}, "人话版布局摘要"),
          readable,
        ),
      ),
    );
    const main = el("div", { class: "mle-main" },
      el("div", { class: "mle-canvas-col" }, frame, status),
      side,
    );
    host.append(toolbar, main);

    function setStatus(text, error) {
      status.textContent = text;
      status.classList.toggle("error", Boolean(error));
    }

    function currentSize() {
      const root = appRoot();
      const wInput = root.querySelector(`#${tab}_width input`);
      const hInput = root.querySelector(`#${tab}_height input`);
      state.w = readNumber(wInput, state.w || 1024);
      state.h = readNumber(hInput, state.h || state.w || 1024);
    }

    function normalized(box) {
      return {
        x1: Math.min(box.x1, box.x2), y1: Math.min(box.y1, box.y2),
        x2: Math.max(box.x1, box.x2), y2: Math.max(box.y1, box.y2),
      };
    }

    function draftObject() {
      const summaryValue = { caption: state.summary.trim() };
      const metaValue = state.meta.split(",").map((item) => item.trim()).filter(Boolean);
      if (metaValue.length) summaryValue.meta = metaValue;
      if (state.canvas !== "single") summaryValue.canvas = state.canvas;
      const items = state.boxes.map((box) => {
        const b = normalized(box);
        const item = {
          label: box.type,
          caption: box.caption || "",
          bbox: [Math.round(b.x1), Math.round(b.y1), Math.round(b.x2), Math.round(b.y2)],
        };
        if (box.type === "text") item.text = box.text || "";
        if (box.palette && box.palette.length) item.palette = box.palette;
        return item;
      });
      return {
        width: state.w, height: state.h,
        bbox_coord_space: "abs", bbox_format: "xyxy",
        summary: summaryValue, items,
      };
    }

    function finalObject() {
      const draft = draftObject();
      const items = draft.items.map((item) => {
        const [x1, y1, x2, y2] = item.bbox;
        const result = { label: item.label, caption: item.caption };
        if (item.label === "text") result.text = item.text || "";
        result.bbox = [
          Math.floor(y1 - state.h / 2), Math.floor(x1 - state.w / 2),
          Math.ceil(y2 - state.h / 2), Math.ceil(x2 - state.w / 2),
        ];
        if (item.palette) result.palette = item.palette;
        return result;
      });
      return {
        summary: draft.summary, items,
        bbox_coord_space: "abs_centralized", bbox_format: "yxyx",
      };
    }

    function validateForWrite() {
      if (!state.boxes.length) {
        setStatus("还没有框：先在画布上拖一个。", true);
        return false;
      }
      const bad = state.boxes.findIndex((box) => {
        const b = normalized(box);
        return b.x2 - b.x1 < MIN_BOX || b.y2 - b.y1 < MIN_BOX
          || (box.type === "text" && !String(box.text || "").trim());
      });
      if (bad >= 0) {
        const box = state.boxes[bad];
        setStatus(`第 ${bad + 1} 个框不完整：请检查大小${box.type === "text" ? "和文字内容" : ""}。`, true);
        return false;
      }
      return true;
    }

    function updateFrame() {
      currentSize();
      frame.style.aspectRatio = `${state.w} / ${state.h}`;
      sizeLabel.textContent = `${state.w} × ${state.h}`;
      for (const box of state.boxes) clampBox(box, state.w, state.h);
    }

    function readableLayout() {
      const lines = [`画布：${state.w} × ${state.h}`];
      if (state.summary.trim()) lines.push(`整体：${state.summary.trim()}`);
      state.boxes.forEach((box, index) => {
        const content = box.type === "text"
          ? [box.caption, box.text].filter(Boolean).join(" / ")
          : (box.caption || "");
        lines.push(
          `${index + 1}. ${TYPE_NAMES[box.type] || "物体"}：`
          + (content || "（未描述）"),
        );
      });
      return lines.join("\n");
    }

    function renderList() {
      list.innerHTML = "";
      countBadge.textContent = String(state.boxes.length);
      readable.textContent = readableLayout();
      if (!state.boxes.length) {
        list.append(el("div", { class: "mle-empty" }, "还没有框，拖拽画布或点「添加」。"));
        return;
      }
      state.boxes.forEach((box, index) => {
        const row = el("div", { class: `mle-row-item${index === state.selected ? " selected" : ""}` },
          el("button", {
            type: "button", class: "mle-row-main", title: "单击选中，双击编辑",
            onclick: () => {
              const now = performance.now();
              if (state.lastRowClick && state.lastRowClick.index === index && now - state.lastRowClick.time < 350) {
                state.lastRowClick = null;
                openEditor(index);
                return;
              }
              state.lastRowClick = { index, time: now };
              state.selected = index;
              draw();
            },
          },
            el("span", { class: "mle-row-num", style: `background:${COLORS[box.type] || COLORS.object}` }, String(index + 1)),
            el("span", {
              class: "mle-row-text",
              title: box.caption || box.text || "（未描述）",
            }, `${TYPE_NAMES[box.type] || "物体"} · ${box.caption || box.text || "（未描述）"}`),
          ),
          el("button", {
            type: "button", class: "mle-row-edit", title: "编辑这个框的提示词",
            onclick: () => openEditor(index),
          }, "编辑"),
          el("button", {
            type: "button", class: "mle-row-del", title: "删除",
            onclick: () => { state.boxes.splice(index, 1); state.selected = -1; draw(); },
          }, "×"),
        );
        list.append(row);
      });
    }

    function draw() {
      updateFrame();
      bg.style.display = state.hasBackground ? "block" : "none";
      boxes.innerHTML = "";
      hint.style.display = state.boxes.length ? "none" : "block";
      state.boxes.forEach((box, index) => {
        const b = normalized(box);
        const left = (b.x1 / state.w) * 100;
        const top = (b.y1 / state.h) * 100;
        const width = ((b.x2 - b.x1) / state.w) * 100;
        const height = ((b.y2 - b.y1) / state.h) * 100;
        const color = COLORS[box.type] || COLORS.object;
        const node = el("div", {
          class: `mle-box${index === state.selected ? " selected" : ""}`,
          style: `left:${left}%;top:${top}%;width:${width}%;height:${height}%;border-color:${color}`,
          "data-index": String(index),
        },
          el("span", { class: "mle-tag", style: `background:${color}` },
            `${index + 1} ${TYPE_NAMES[box.type] || "物体"}${box.caption ? ` · ${box.caption}` : box.text ? ` · ${box.text}` : ""}`),
        );
        node.addEventListener("pointerdown", (event) => {
          event.stopPropagation();
          event.preventDefault();
          state.selected = index;
          startDrag("move", event, box, index);
          draw();
        });
        node.addEventListener("dblclick", (event) => {
          event.stopPropagation();
          openEditor(index);
        });
        if (index === state.selected) {
          for (const corner of ["nw", "ne", "sw", "se"]) {
            const handle = el("div", {
              class: `mle-handle ${corner}`,
              style: `border-color:${color}`,
            });
            handle.addEventListener("pointerdown", (event) => {
              event.stopPropagation();
              event.preventDefault();
              startDrag(corner, event, box, index);
            });
            node.append(handle);
          }
        }
        boxes.append(node);
      });
      summary.value = state.summary;
      meta.value = state.meta;
      canvas.value = state.canvas;
      renderList();
    }

    function point(event) {
      const rect = frame.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(state.w, ((event.clientX - rect.left) / rect.width) * state.w)),
        y: Math.max(0, Math.min(state.h, ((event.clientY - rect.top) / rect.height) * state.h)),
      };
    }

    function startDrag(mode, event, box, index) {
      const p = point(event);
      state.drag = {
        mode, box, index,
        moved: false,
        clientX: event.clientX, clientY: event.clientY,
        startX: p.x, startY: p.y,
        original: { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 },
      };
      window.addEventListener("pointermove", onDragMove);
      window.addEventListener("pointerup", onDragEnd, { once: true });
    }

    function onDragMove(event) {
      if (!state.drag) return;
      if (Math.abs(event.clientX - state.drag.clientX) + Math.abs(event.clientY - state.drag.clientY) > 4) {
        state.drag.moved = true;
      }
      const p = point(event);
      const { mode, box, original, startX, startY } = state.drag;
      const dx = p.x - startX;
      const dy = p.y - startY;
      if (mode === "move") {
        const width = original.x2 - original.x1;
        const height = original.y2 - original.y1;
        const x1 = Math.max(0, Math.min(state.w - width, original.x1 + dx));
        const y1 = Math.max(0, Math.min(state.h - height, original.y1 + dy));
        box.x1 = x1;
        box.x2 = x1 + width;
        box.y1 = y1;
        box.y2 = y1 + height;
      } else if (mode === "new") {
        box.x1 = Math.min(startX, p.x);
        box.x2 = Math.max(startX, p.x);
        box.y1 = Math.min(startY, p.y);
        box.y2 = Math.max(startY, p.y);
      } else {
        box.x1 = original.x1;
        box.y1 = original.y1;
        box.x2 = original.x2;
        box.y2 = original.y2;
        if (mode.includes("w")) box.x1 = Math.max(0, Math.min(original.x2 - MIN_BOX, p.x));
        if (mode.includes("e")) box.x2 = Math.min(state.w, Math.max(original.x1 + MIN_BOX, p.x));
        if (mode.includes("n")) box.y1 = Math.max(0, Math.min(original.y2 - MIN_BOX, p.y));
        if (mode.includes("s")) box.y2 = Math.min(state.h, Math.max(original.y1 + MIN_BOX, p.y));
      }
      box.x1 = Math.max(0, Math.min(state.w, box.x1));
      box.x2 = Math.max(0, Math.min(state.w, box.x2));
      box.y1 = Math.max(0, Math.min(state.h, box.y1));
      box.y2 = Math.max(0, Math.min(state.h, box.y2));
      draw();
    }

    function onDragEnd() {
      if (!state.drag) return;
      const { mode, box, index, moved } = state.drag;
      state.drag = null;
      window.removeEventListener("pointermove", onDragMove);
      const b = normalized(box);
      if (mode === "new" && (b.x2 - b.x1 < MIN_BOX || b.y2 - b.y1 < MIN_BOX)) {
        state.boxes = state.boxes.filter((item) => item !== box);
        state.selected = -1;
        draw();
        return;
      }
      // 原生 dblclick 在 pointerdown 重建节点后不会触发，所以在 pointerup
      // 里自己做双击判定：同一个框、350ms 内、没有拖动。
      if (mode === "move" && !moved) {
        const now = performance.now();
        if (state.lastBoxClick && state.lastBoxClick.index === index && now - state.lastBoxClick.time < 350) {
          state.lastBoxClick = null;
          draw();
          openEditor(index);
          return;
        }
        state.lastBoxClick = { index, time: now };
      }
      draw();
    }

    function openEditor(index) {
      const box = state.boxes[index];
      if (!box || state.editing >= 0) return;
      state.editing = index;
      const shade = el("div", { class: `mle-modal-shade${host.classList.contains("mle-dark") ? " mle-dark" : ""}` });
      const type = el("select", { class: "mle-modal-type" },
        el("option", { value: "object" }, "物体 object"),
        el("option", { value: "character" }, "角色 character"),
        el("option", { value: "text" }, "文字 text"),
      );
      type.value = box.type;
      const caption = el("textarea", { class: "mle-modal-caption", rows: "3", placeholder: "这个框里是什么：外观、颜色、材质、动作、构图……" });
      caption.value = box.caption || "";
      const text = el("textarea", { class: "mle-modal-text", rows: "2", placeholder: "要写进画面的原文（文字框必填）" });
      text.value = box.text || "";
      const textWrap = el("label", { class: "mle-modal-field mle-modal-text-wrap" }, "精确文字", text);
      const updateTextVisibility = () => { textWrap.style.display = type.value === "text" ? "flex" : "none"; };
      type.addEventListener("change", updateTextVisibility);
      updateTextVisibility();
      const dialog = el("div", { class: "mle-modal" },
        el("h3", {}, `编辑第 ${index + 1} 个框`),
        el("label", { class: "mle-modal-field" }, "类型", type),
        el("label", { class: "mle-modal-field" }, "这个框的提示词 / 描述", caption),
        textWrap,
        el("div", { class: "mle-modal-actions" },
          el("button", { type: "button", class: "mle-btn danger", onclick: () => close(true) }, "删除"),
          el("span", { class: "mle-spacer" }),
          el("button", { type: "button", class: "mle-btn", onclick: () => close(false) }, "取消"),
          el("button", { type: "button", class: "mle-btn primary", onclick: () => save() }, "保存"),
        ),
      );
      shade.append(dialog);
      document.body.append(shade);
      dialog.addEventListener("pointerdown", (event) => event.stopPropagation());
      shade.addEventListener("pointerdown", (event) => { if (event.target === shade) close(false); });
      const onKey = (event) => { if (event.key === "Escape") close(false); };
      window.addEventListener("keydown", onKey);
      function close(remove) {
        window.removeEventListener("keydown", onKey);
        if (remove) {
          state.boxes.splice(index, 1);
          state.selected = -1;
        }
        shade.remove();
        state.editing = -1;
        draw();
      }
      function save() {
        const value = type.value;
        if (value === "text" && !text.value.trim()) {
          text.focus();
          return;
        }
        box.type = value;
        box.caption = caption.value.trim();
        box.text = text.value;
        close(false);
      }
      caption.focus();
    }

    function openImportDialog() {
      if (state.importing) return;
      state.importing = true;
      const shade = el("div", { class: `mle-modal-shade${host.classList.contains("mle-dark") ? " mle-dark" : ""}` });
      const area = el("textarea", {
        class: "mle-import-text", rows: "12",
        placeholder: '粘贴 Layout JSON，例如：\n{\n  "width": 1024,\n  "height": 1024,\n  "bbox_coord_space": "abs",\n  "bbox_format": "xyxy",\n  "summary": {"caption": "..."},\n  "items": [...]\n}',
      });
      const error = el("div", { class: "mle-import-error" });
      const dialog = el("div", { class: "mle-modal mle-import-modal" },
        el("h3", {}, "导入 Layout JSON"),
        el("div", { class: "mle-modal-help" },
          "支持草稿（abs + xyxy）和最终格式（abs_centralized + yxyx）；可以带 Markdown 代码块。"
          + "导入后会按当前画布尺寸缩放到画布上。",
        ),
        area,
        error,
        el("div", { class: "mle-modal-actions" },
          el("button", { type: "button", class: "mle-btn", onclick: () => close() }, "取消"),
          el("span", { class: "mle-spacer" }),
          el("button", { type: "button", class: "mle-btn primary", onclick: () => apply() }, "导入画布"),
        ),
      );
      shade.append(dialog);
      document.body.append(shade);
      dialog.addEventListener("pointerdown", (event) => event.stopPropagation());
      shade.addEventListener("pointerdown", (event) => { if (event.target === shade) close(); });
      const onKey = (event) => { if (event.key === "Escape") close(); };
      window.addEventListener("keydown", onKey);
      function close() {
        window.removeEventListener("keydown", onKey);
        shade.remove();
        state.importing = false;
      }
      function showError(message) {
        error.textContent = message;
      }
      function apply() {
        const raw = area.value.trim();
        if (!raw) {
          showError("请先粘贴 JSON。");
          return;
        }
        const json = findJson(raw);
        if (!json) {
          showError("没有找到完整的 JSON 对象。");
          return;
        }
        try {
          const data = JSON.parse(json);
          loadLayoutObject(data, state.w, state.h);
          close();
          setStatus(`已导入 ${state.boxes.length} 个框；点「写入正向提示词」才会用于生成。`);
        } catch (err) {
          showError(`导入失败：${err.message || err}`);
        }
      }
      area.addEventListener("input", () => { error.textContent = ""; });
      area.focus();
    }

    function addBox() {
      const width = Math.min(320, state.w * 0.35);
      const height = Math.min(220, state.h * 0.25);
      state.boxes.push({
        x1: (state.w - width) / 2, y1: (state.h - height) / 2,
        x2: (state.w + width) / 2, y2: (state.h + height) / 2,
        type: "object", caption: "", text: "", palette: [],
      });
      state.selected = state.boxes.length - 1;
      draw();
      openEditor(state.selected);
    }

    function loadLayoutObject(data, sourceWidth, sourceHeight) {
      if (!data || !Array.isArray(data.items)) throw new Error("不是 MOSS Layout（缺少 items）");
      const final = data.bbox_coord_space === "abs_centralized" && data.bbox_format === "yxyx";
      const draft = data.bbox_coord_space === "abs" && data.bbox_format === "xyxy";
      if (!final && !draft) {
        throw new Error("需要 abs + xyxy（草稿）或 abs_centralized + yxyx（最终格式）");
      }
      // 草稿的 bbox 是草稿自己 width/height 下的左上角像素；最终格式的 bbox
      // 是中心坐标，必须按当前画布解释。
      const baseW = final
        ? (Number(sourceWidth || state.w) || state.w)
        : (Number(data.width || sourceWidth || state.w) || state.w);
      const baseH = final
        ? (Number(sourceHeight || state.h) || state.h)
        : (Number(data.height || sourceHeight || state.h) || state.h);
      const sx = state.w / baseW;
      const sy = state.h / baseH;
      const characterSet = new Set(
        String(data.character_boxes || "")
          .match(/\d+/g)
          ?.map((value) => Number(value)) || [],
      );
      state.boxes = data.items.map((item, itemIndex) => {
        const bbox = Array.isArray(item.bbox) ? item.bbox : [0, 0, 0, 0];
        let x1, y1, x2, y2;
        if (final) {
          y1 = (Number(bbox[0]) + baseH / 2) * sy;
          x1 = (Number(bbox[1]) + baseW / 2) * sx;
          y2 = (Number(bbox[2]) + baseH / 2) * sy;
          x2 = (Number(bbox[3]) + baseW / 2) * sx;
        } else {
          x1 = Number(bbox[0]) * sx;
          y1 = Number(bbox[1]) * sy;
          x2 = Number(bbox[2]) * sx;
          y2 = Number(bbox[3]) * sy;
        }
        const rawLabel = String(item.label || item.type || "object").toLowerCase();
        const isCharacter = rawLabel === "character" || characterSet.has(itemIndex + 1);
        return clampBox({
          x1, y1, x2, y2,
          type: rawLabel === "text" ? "text" : isCharacter ? "character" : "object",
          caption: String(item.caption ?? item.desc ?? ""),
          text: String(item.text ?? ""),
          palette: Array.isArray(item.palette) ? item.palette : [],
        }, state.w, state.h);
      });
      state.summary = String((data.summary && data.summary.caption) || "");
      state.meta = Array.isArray(data.summary && data.summary.meta) ? data.summary.meta.join(", ") : "";
      state.canvas = String((data.summary && data.summary.canvas) || "single");
      state.selected = -1;
      draw();
      if (baseW !== state.w || baseH !== state.h) {
        setStatus(`已按 ${baseW}×${baseH} 的布局载入，并缩放到当前 ${state.w}×${state.h}。`);
      }
    }

    function readFromPrompt() {
      const prompt = getTextarea(`${tab}_prompt`);
      const json = findJson(prompt);
      if (!json) {
        setStatus("正向提示词里没有找到 Layout JSON。", true);
        return;
      }
      try {
        loadLayoutObject(JSON.parse(json), state.w, state.h);
        setStatus("已从正向提示词读入布局。");
      } catch (error) {
        setStatus(`读取失败：${error.message || error}`, true);
      }
    }

    function writeToPrompt() {
      if (!validateForWrite()) return;
      const prompt = getTextarea(`${tab}_prompt`);
      const tags = tagsAfterJson(prompt);
      const json = JSON.stringify(finalObject());
      const value = tags ? `${json} ${tags}` : json;
      setTextarea(`moss_layout_${tab}_json`, json);
      if (!setTextarea(`${tab}_prompt`, value)) {
        setStatus("找不到正向提示词输入框，请手动复制高级里的 JSON。", true);
        return;
      }
      setStatus(`已写入正向提示词（${state.boxes.length} 个框，${state.w}×${state.h}）。`);
    }

    function writeDraft() {
      const json = JSON.stringify(draftObject(), null, 2);
      if (setTextarea(`moss_layout_${tab}_json`, json)) setStatus("草稿 JSON 已写入下方文本框。");
      else setStatus("找不到 Layout JSON 文本框。", true);
    }

    function writeCompiled() {
      if (!validateForWrite()) return;
      const json = JSON.stringify(finalObject());
      if (setTextarea(`moss_layout_${tab}_json`, json)) setStatus("最终 JSON 已写入下方文本框，可直接粘贴到正向提示词。");
      else setStatus("找不到 Layout JSON 文本框。", true);
    }

    function clearAll() {
      if (!state.boxes.length && !state.summary) return;
      if (!window.confirm("清空当前布局（框和描述）？")) return;
      state.boxes = [];
      state.selected = -1;
      state.summary = "";
      state.meta = "";
      state.canvas = "single";
      draw();
      setStatus("已清空。");
    }

    function attachBackground() {
      const image = appRoot().querySelector(`#${tab}_image img, #${tab}_image canvas`);
      if (!image) {
        setStatus("当前标签页没有输入图。", true);
        return;
      }
      if (image.tagName === "IMG" && image.src) {
        bg.src = image.src;
        state.hasBackground = true;
        draw();
        setStatus("已使用当前输入图作为背景。");
      } else if (image.tagName === "CANVAS") {
        bg.src = image.toDataURL("image/png");
        state.hasBackground = true;
        draw();
        setStatus("已使用当前输入图作为背景。");
      }
    }

    // Events -----------------------------------------------------------------
    frame.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".mle-box")) return;
      const p = point(event);
      const box = {
        x1: p.x, y1: p.y, x2: p.x, y2: p.y,
        type: "object", caption: "", text: "", palette: [],
      };
      state.boxes.push(box);
      state.selected = state.boxes.length - 1;
      state.drag = {
        mode: "new", box, moved: false,
        clientX: event.clientX, clientY: event.clientY,
        startX: p.x, startY: p.y, original: { ...box },
      };
      window.addEventListener("pointermove", onDragMove);
      window.addEventListener("pointerup", onDragEnd, { once: true });
      draw();
    });
    importButton.addEventListener("click", openImportDialog);
    readButton.addEventListener("click", readFromPrompt);
    writeButton.addEventListener("click", writeToPrompt);
    draftButton.addEventListener("click", writeDraft);
    compileButton.addEventListener("click", writeCompiled);
    clearButton.addEventListener("click", clearAll);
    addButton.addEventListener("click", addBox);
    summary.addEventListener("input", () => {
      state.summary = summary.value;
      readable.textContent = readableLayout();
    });
    meta.addEventListener("input", () => { state.meta = meta.value; });
    canvas.addEventListener("change", () => { state.canvas = canvas.value; });
    file.addEventListener("change", () => {
      const selected = file.files && file.files[0];
      if (!selected) return;
      const reader = new FileReader();
      reader.onload = () => {
        bg.src = String(reader.result || "");
        state.hasBackground = true;
        draw();
        setStatus(`已载入背景图：${selected.name}`);
      };
      reader.readAsDataURL(selected);
    });

    // Size changes from the native sliders / number fields.
    const root = appRoot();
    for (const id of [`${tab}_width`, `${tab}_height`]) {
      const input = root.querySelector(`#${id} input`);
      if (input) {
        input.addEventListener("input", () => { draw(); });
        input.addEventListener("change", () => { draw(); });
      }
    }

    // img2img background shortcut.
    if (tab === "img2img") {
      const useImageButton = el("button", { type: "button", class: "mle-btn quiet" }, "用输入图");
      useImageButton.title = "把 img2img 当前输入图作为画布背景";
      useImageButton.addEventListener("click", attachBackground);
      toolbarLeft.insertBefore(useImageButton, fileLabel.nextSibling);
    }

    draw();
    setStatus("画框后点「写入正向提示词」。草稿格式会在生成时自动编译。");
  }

  function initAll() {
    const root = appRoot();
    root.querySelectorAll(".moss-layout-editor[data-tab]").forEach((host) => {
      if (host.dataset.mleReady) return;
      host.dataset.mleReady = "1";
      try {
        build(host, host.dataset.tab || "txt2img");
      } catch (error) {
        host.dataset.mleReady = "";
        console.error("[MOSS-Image] Layout editor failed:", error);
      }
    });
  }

  if (typeof onUiLoaded === "function") onUiLoaded(initAll);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(initAll, 300));
  } else {
    setTimeout(initAll, 300);
  }
  // Forge Neo rebuilds tabs on some UI reloads; keep the observer cheap.
  if (typeof MutationObserver !== "undefined") {
    const observer = new MutationObserver(() => {
      if (document.querySelector(".moss-layout-editor[data-tab]:not([data-mle-ready])")) initAll();
    });
    const start = () => observer.observe(document.body, { childList: true, subtree: true });
    if (document.body) start();
    else document.addEventListener("DOMContentLoaded", start);
  }
})();
