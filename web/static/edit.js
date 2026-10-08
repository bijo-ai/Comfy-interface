// LUMOS Edit studio: one image, a history of versions, AI tools (Fix, Restyle, Upscale) and instant
// browser tools (crop, rotate, flip, adjust, filters). AI results land in the Gallery automatically;
// browser edits are drafts until Save.

const ASPECTS = { free: null, "1:1": 1, "4:5": 4 / 5, "3:2": 3 / 2, "16:9": 16 / 9, "9:16": 9 / 16 };
const NEUTRAL = { brightness: 1, contrast: 1, saturation: 1, warmth: 0, grayscale: 0, sepia: 0 };
const FILTERS = {
  vivid: { label: "Vivid", contrast: 1.12, saturation: 1.35, brightness: 1.03 },
  matte: { label: "Matte", contrast: 0.86, saturation: 0.9, brightness: 1.06 },
  noir: { label: "Noir", grayscale: 1, contrast: 1.25, brightness: 0.98 },
  warmfilm: { label: "Warm film", warmth: 0.35, contrast: 1.05, saturation: 1.08, sepia: 0.15 },
  cool: { label: "Cool", warmth: -0.3, saturation: 0.95, brightness: 1.02 },
  fade: { label: "Fade", contrast: 0.8, brightness: 1.08, saturation: 0.82 },
};
const TOOL_TITLES = {
  fix: "Fix part", restyle: "Restyle", upscale: "Upscale ×2", crop: "Crop & rotate", adjust: "Adjust", filters: "Filters",
};
const UPSCALE_MAX_SIDE = 768;
const MASK_COLOR = "rgb(242, 181, 68)";

export function initEdit(deps) {
  const { api, toast, imageUrl, runJob, fitSize, getModelKey, getStyles, showTab, onNewImages, isBusy } = deps;
  const $ = (selector) => document.querySelector(selector);
  const el = {
    view: $("#edit-view"), empty: $("#edit-empty"), workspace: $("#edit-workspace"), pick: $("#edit-pick"), file: $("#edit-file"),
    tools: [...document.querySelectorAll("#edit-view .tool")], panels: [...document.querySelectorAll("#edit-view [data-panel]")],
    title: $("#edit-tool-title"), save: $("#edit-save"), canvas: $("#edit-canvas"), frame: $("#edit-frame"),
    img: $("#edit-img"), before: $("#edit-before"), tint: $("#edit-tint"), mask: $("#edit-mask"), crop: $("#edit-crop"),
    liveHost: $("#edit-live-host"), history: $("#edit-history"), compare: $("#edit-compare"), compareRange: $("#edit-compare-range"),
    fit: $("#edit-fit"), actual: $("#edit-100"), zoomLabel: $("#edit-zoom"),
    // tool panels
    fixPrompt: $("#edit-fix-prompt"), brush: $("#edit-brush"), brushValue: $("#edit-brush-value"),
    paint: $("#edit-paint"), erase: $("#edit-erase"), clearMask: $("#edit-clear-mask"), fixGo: $("#edit-fix-go"),
    restylePrompt: $("#edit-restyle-prompt"), restyleStrength: $("#edit-restyle-strength"), restyleStrengthValue: $("#edit-restyle-strength-value"),
    restyleStyle: $("#edit-restyle-style"), restyleGo: $("#edit-restyle-go"),
    upscaleInfo: $("#edit-upscale-info"), upscaleGo: $("#edit-upscale-go"),
    aspects: [...document.querySelectorAll("#edit-view [data-aspect]")], cropApply: $("#edit-crop-apply"),
    rotateLeft: $("#edit-rotate-left"), rotateRight: $("#edit-rotate-right"), flipH: $("#edit-flip-h"), flipV: $("#edit-flip-v"),
    sliders: [...document.querySelectorAll("#edit-view [data-adjust]")], adjustReset: $("#edit-adjust-reset"), adjustApply: $("#edit-adjust-apply"),
    filterGrid: $("#edit-filter-grid"), filterApply: $("#edit-filter-apply"),
  };

  const session = { versions: [], current: -1, tool: "restyle", adjust: { ...NEUTRAL }, filter: null, crop: null, aspect: null };
  const view = { scale: 1, x: 0, y: 0, panning: null };
  const brush = { erasing: false, drawing: false, last: null };

  const current = () => session.versions[session.current];
  const sourceFields = (version) => (version.name ? { source_name: version.name } : { source_id: version.sourceId });

  // --- sessions and versions ----------------------------------------------------------------

  function open(image, tool = "restyle") {
    session.versions = [];
    session.current = -1;
    addVersion({ label: "Original", name: image.name, url: imageUrl(image.name), width: image.width, height: image.height, params: image.params });
    selectTool(tool);
    if (image.params?.prompt) {
      el.restylePrompt.value = image.params.prompt;
      el.fixPrompt.value = image.params.prompt;
    }
    showTab("edit");
  }

  async function openFile(file) {
    if (!file?.type.startsWith("image/")) {
      toast("That file isn't an image.", "error");
      return;
    }
    try {
      const data = await api("/api/sources", { method: "POST", headers: { "Content-Type": file.type }, body: file });
      session.versions = [];
      session.current = -1;
      addVersion({ label: "Original", sourceId: data.id, url: `/api/sources/${data.id}`, width: data.width, height: data.height, params: null });
      el.restylePrompt.value = "";
      el.fixPrompt.value = "";
      selectTool("restyle");
      showTab("edit");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  function addVersion(version) {
    session.versions.push(version);
    setCurrent(session.versions.length - 1);
  }

  function setCurrent(index) {
    session.current = index;
    const version = current();
    el.empty.hidden = true;
    el.workspace.hidden = false;
    el.img.onload = () => {
      el.frame.style.width = `${el.img.naturalWidth}px`;
      el.frame.style.height = `${el.img.naturalHeight}px`;
      for (const canvas of [el.mask]) {
        canvas.width = el.img.naturalWidth;
        canvas.height = el.img.naturalHeight;
      }
      fitView();
      resetCrop();
    };
    el.img.src = version.url;
    el.before.src = session.versions[0].url;
    resetPreview();
    renderHistory();
    renderPanels();
  }

  function renderHistory() {
    el.history.replaceChildren(...session.versions.map((version, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "version";
      button.setAttribute("aria-current", String(index === session.current));
      button.title = version.sourceId && !version.saved ? `${version.label} (draft, not saved yet)` : version.label;
      const thumb = document.createElement("img");
      thumb.alt = "";
      thumb.src = version.name ? imageUrl(version.name, "thumbs") : version.url;
      const label = document.createElement("span");
      label.textContent = `${index + 1}. ${version.label}${version.sourceId && !version.saved ? " •" : ""}`;
      button.append(thumb, label);
      button.addEventListener("click", () => setCurrent(index));
      return button;
    }));
    el.history.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  function renderPanels() {
    const version = current();
    if (!version) return;
    const draft = Boolean(version.sourceId && !version.saved);
    el.save.disabled = !draft || isBusy();
    el.save.title = draft ? "Save this version to the Gallery" : "This version is already in the Gallery";
    const tooBig = Math.max(version.width, version.height) > UPSCALE_MAX_SIDE;
    el.upscaleGo.disabled = tooBig || isBusy();
    el.upscaleInfo.textContent = tooBig
      ? `This image is ${version.width}×${version.height}: already large, so upscaling it would overload your GPU.`
      : `${version.width}×${version.height} → ${version.width * 2}×${version.height * 2}, with real added detail.`;
    for (const button of [el.fixGo, el.restyleGo]) button.disabled = isBusy();
  }

  function selectTool(tool) {
    if (session.tool === "adjust" || session.tool === "filters") resetPreview();
    session.tool = tool;
    for (const button of el.tools) button.setAttribute("aria-pressed", String(button.dataset.tool === tool));
    for (const panel of el.panels) panel.hidden = panel.dataset.panel !== tool;
    el.title.textContent = TOOL_TITLES[tool];
    el.mask.hidden = tool !== "fix";
    el.crop.hidden = tool !== "crop";
    if (tool === "crop") resetCrop();
    renderPanels();
  }

  // --- view: fit, 100%, wheel zoom, drag to pan ------------------------------------------------

  function applyView() {
    el.frame.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
    el.frame.style.setProperty("--scale", String(view.scale)); // keeps outlines crisp at any zoom
    el.zoomLabel.textContent = `${Math.round(view.scale * 100)}%`;
  }

  function fitView() {
    const box = el.canvas.getBoundingClientRect();
    const width = el.img.naturalWidth || 1;
    const height = el.img.naturalHeight || 1;
    view.scale = Math.min((box.width - 48) / width, (box.height - 48) / height, 4);
    view.x = (box.width - width * view.scale) / 2;
    view.y = (box.height - height * view.scale) / 2;
    applyView();
  }

  function zoomAt(scale, clientX, clientY) {
    const box = el.canvas.getBoundingClientRect();
    const px = clientX - box.left;
    const py = clientY - box.top;
    const next = Math.min(Math.max(scale, 0.1), 8);
    view.x = px - ((px - view.x) / view.scale) * next;
    view.y = py - ((py - view.y) / view.scale) * next;
    view.scale = next;
    applyView();
  }

  function imagePoint(event) {
    const rect = el.img.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) / rect.width) * el.img.naturalWidth,
      ((event.clientY - rect.top) / rect.height) * el.img.naturalHeight,
    ];
  }

  // --- instant tools: rendered with a canvas, saved as drafts -----------------------------------

  function filterString(values) {
    return [
      `brightness(${values.brightness})`, `contrast(${values.contrast})`, `saturate(${values.saturation})`,
      values.grayscale ? `grayscale(${values.grayscale})` : "", values.sepia ? `sepia(${values.sepia})` : "",
    ].join(" ").trim();
  }

  function tintStyle(warmth) {
    if (!warmth) return null;
    return warmth > 0 ? `rgba(255, 140, 40, ${warmth * 0.55})` : `rgba(60, 130, 255, ${-warmth * 0.55})`;
  }

  function previewLook(values) {
    el.img.style.filter = filterString(values);
    const tint = tintStyle(values.warmth);
    el.tint.hidden = !tint;
    if (tint) el.tint.style.background = tint;
  }

  function resetPreview() {
    session.adjust = { ...NEUTRAL };
    session.filter = null;
    for (const slider of el.sliders) slider.value = slider.dataset.neutral;
    for (const chip of el.filterGrid.children) chip.setAttribute("aria-checked", "false");
    el.img.style.filter = "";
    el.tint.hidden = true;
  }

  async function loadCurrentImage() {
    const picture = new Image();
    picture.src = current().url;
    await picture.decode();
    return picture;
  }

  async function bake(label, width, height, draw) {
    const base = current();
    const picture = await loadCurrentImage();
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width(picture));
    canvas.height = Math.round(height(picture));
    const ctx = canvas.getContext("2d");
    draw(ctx, picture, canvas);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    try {
      const data = await api("/api/sources", { method: "POST", headers: { "Content-Type": "image/png" }, body: blob });
      addVersion({
        label, sourceId: data.id, url: `/api/sources/${data.id}`, width: data.width, height: data.height,
        params: base.params, parentName: base.name ?? base.parentName ?? null,
      });
    } catch (error) {
      toast(error.message, "error");
    }
  }

  function bakeLook(label, values) {
    return bake(label, (p) => p.naturalWidth, (p) => p.naturalHeight, (ctx, picture, canvas) => {
      ctx.filter = filterString(values);
      ctx.drawImage(picture, 0, 0);
      ctx.filter = "none";
      const tint = tintStyle(values.warmth);
      if (tint) {
        ctx.globalCompositeOperation = "soft-light";
        ctx.fillStyle = tint;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.globalCompositeOperation = "source-over";
      }
    });
  }

  function rotate(quarterTurns) {
    const turn = ((quarterTurns % 4) + 4) % 4;
    return bake(turn === 1 ? "Rotated right" : "Rotated left",
      (p) => (turn % 2 ? p.naturalHeight : p.naturalWidth), (p) => (turn % 2 ? p.naturalWidth : p.naturalHeight),
      (ctx, picture, canvas) => {
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate((turn * Math.PI) / 2);
        ctx.drawImage(picture, -picture.naturalWidth / 2, -picture.naturalHeight / 2);
      });
  }

  function flip(horizontal) {
    return bake(horizontal ? "Flipped" : "Flipped vertically", (p) => p.naturalWidth, (p) => p.naturalHeight,
      (ctx, picture, canvas) => {
        ctx.translate(horizontal ? canvas.width : 0, horizontal ? 0 : canvas.height);
        ctx.scale(horizontal ? -1 : 1, horizontal ? 1 : -1);
        ctx.drawImage(picture, 0, 0);
      });
  }

  // crop box in image pixels; drag on the image to draw one, drag inside it to move it
  function resetCrop() {
    const width = el.img.naturalWidth;
    const height = el.img.naturalHeight;
    if (!width) return;
    session.crop = constrain({ x: 0, y: 0, w: width, h: height }, "center");
    drawCrop();
  }

  function constrain(rect, anchor = "start") {
    const width = el.img.naturalWidth;
    const height = el.img.naturalHeight;
    let { x, y, w, h } = rect;
    if (session.aspect) {
      if (anchor === "center") {
        w = Math.min(width, height * session.aspect);
        h = w / session.aspect;
        x = (width - w) / 2;
        y = (height - h) / 2;
      } else {
        h = w / session.aspect;
        if (y + h > height) { h = height - y; w = h * session.aspect; }
      }
    }
    w = Math.max(16, Math.min(w, width));
    h = Math.max(16, Math.min(h, height));
    x = Math.min(Math.max(0, x), width - w);
    y = Math.min(Math.max(0, y), height - h);
    return { x, y, w, h };
  }

  function drawCrop() {
    const { x, y, w, h } = session.crop;
    Object.assign(el.crop.style, { left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px` });
    el.cropApply.textContent = `Apply crop · ${Math.round(w)}×${Math.round(h)}`;
  }

  function cropNow() {
    const { x, y, w, h } = session.crop;
    const width = Math.round(w);
    const height = Math.round(h);
    return bake(`Cropped ${width}×${height}`, () => width, () => height, (ctx, picture) => {
      ctx.drawImage(picture, Math.round(x), Math.round(y), width, height, 0, 0, width, height);
    });
  }

  // --- AI tools: they run as jobs and land in the Gallery ------------------------------------------

  async function runAiTool(path, body, label) {
    const version = current();
    const max = deps.liveMaxSide(path === "/api/upscale");
    const [width, height] = path === "/api/upscale" ? [version.width * 2, version.height * 2] : fitSize(version.width, version.height, max);
    await runJob(path, body, width, height, {
      liveHost: el.liveHost,
      onDone: (event) => {
        const image = event.images?.[0] ?? event.image;
        if (!image) {
          toast("The result couldn't be found in ComfyUI's output folder.", "error");
          return;
        }
        onNewImages();
        addVersion({ label, name: image.name, url: imageUrl(image.name), width: image.width, height: image.height, params: image.params });
      },
    });
    renderPanels();
  }

  function hasMask() {
    const { data } = el.mask.getContext("2d").getImageData(0, 0, el.mask.width, el.mask.height);
    for (let i = 3; i < data.length; i += 4) if (data[i]) return true;
    return false;
  }

  function maskBlob() {
    const out = document.createElement("canvas");
    out.width = el.mask.width;
    out.height = el.mask.height;
    const ctx = out.getContext("2d");
    const source = el.mask.getContext("2d").getImageData(0, 0, out.width, out.height);
    const pixels = ctx.createImageData(out.width, out.height);
    for (let i = 0; i < source.data.length; i += 4) {
      const value = source.data[i + 3] ? 255 : 0;
      pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = value;
      pixels.data[i + 3] = 255;
    }
    ctx.putImageData(pixels, 0, 0);
    return new Promise((resolve) => out.toBlob(resolve, "image/png"));
  }

  async function fixNow() {
    const prompt = el.fixPrompt.value.trim();
    if (!prompt) return toast("Describe what the picture should show.", "error");
    if (!hasMask()) return toast("Paint over the area you want to change first.", "error");
    let maskId;
    try {
      maskId = (await api("/api/sources", { method: "POST", headers: { "Content-Type": "image/png" }, body: await maskBlob() })).id;
    } catch (error) {
      return toast(error.message, "error");
    }
    await runAiTool("/api/inpaint", { prompt, mask_id: maskId, style: el.restyleStyle.value, ...sourceFields(current()) }, "Fixed");
    el.mask.getContext("2d").clearRect(0, 0, el.mask.width, el.mask.height);
  }

  function restyleNow() {
    const prompt = el.restylePrompt.value.trim();
    if (!prompt) return toast("Describe how the picture should look.", "error");
    return runAiTool("/api/generate", {
      prompt, strength: Number(el.restyleStrength.value), style: el.restyleStyle.value, model: getModelKey(),
      batch: 1, ...sourceFields(current()),
    }, "Restyled");
  }

  function upscaleNow() {
    const version = current();
    const body = version.name ? { name: version.name } : { source_id: version.sourceId, parent_name: version.parentName };
    return runAiTool("/api/upscale", body, "Upscaled ×2");
  }

  async function saveNow() {
    const version = current();
    try {
      const saved = await api("/api/save", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_id: version.sourceId, parent_name: version.parentName }),
      });
      version.saved = true;
      version.name = saved.name;
      onNewImages();
      toast(`Saved to the Gallery as ${saved.name}.`);
      renderHistory();
      renderPanels();
    } catch (error) {
      toast(error.message, "error");
    }
  }

  // --- brush for Fix --------------------------------------------------------------------------

  function strokeTo(point) {
    const ctx = el.mask.getContext("2d");
    const rect = el.img.getBoundingClientRect();
    ctx.globalCompositeOperation = brush.erasing ? "destination-out" : "source-over";
    ctx.strokeStyle = MASK_COLOR;
    ctx.lineWidth = Number(el.brush.value) * (el.img.naturalWidth / rect.width);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    const [x0, y0] = brush.last ?? point;
    ctx.moveTo(x0, y0);
    ctx.lineTo(point[0], point[1]);
    ctx.stroke();
    brush.last = point;
  }

  function setBrushMode(erasing) {
    brush.erasing = erasing;
    el.paint.setAttribute("aria-checked", String(!erasing));
    el.erase.setAttribute("aria-checked", String(erasing));
  }

  // --- wiring --------------------------------------------------------------------------------

  function renderStyleOptions() {
    const styles = getStyles();
    if (!styles.length || el.restyleStyle.options.length > 1) return;
    el.restyleStyle.replaceChildren(...styles.map((style) => new Option(`${style.emoji} ${style.label}`, style.key)));
  }

  el.filterGrid.replaceChildren(...Object.entries(FILTERS).map(([key, filter]) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.setAttribute("role", "radio");
    chip.setAttribute("aria-checked", "false");
    chip.dataset.filter = key;
    chip.textContent = filter.label;
    chip.addEventListener("click", () => {
      session.filter = key;
      for (const other of el.filterGrid.children) other.setAttribute("aria-checked", String(other === chip));
      previewLook({ ...NEUTRAL, ...filter });
    });
    return chip;
  }));

  for (const button of el.tools) button.addEventListener("click", () => selectTool(button.dataset.tool));
  el.pick.addEventListener("click", () => el.file.click());
  el.file.addEventListener("change", () => openFile(el.file.files[0]));
  el.view.addEventListener("dragover", (event) => { event.preventDefault(); el.view.classList.add("dragging"); });
  el.view.addEventListener("dragleave", () => el.view.classList.remove("dragging"));
  el.view.addEventListener("drop", (event) => {
    event.preventDefault();
    el.view.classList.remove("dragging");
    openFile(event.dataTransfer.files[0]);
  });
  el.save.addEventListener("click", saveNow);
  el.fit.addEventListener("click", fitView);
  el.actual.addEventListener("click", () => {
    const box = el.canvas.getBoundingClientRect();
    zoomAt(1, box.left + box.width / 2, box.top + box.height / 2);
  });
  el.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomAt(view.scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12), event.clientX, event.clientY);
  }, { passive: false });
  el.compare.addEventListener("click", () => {
    const on = el.before.hidden;
    el.before.hidden = !on;
    el.compareRange.hidden = !on;
    el.compare.setAttribute("aria-pressed", String(on));
    el.before.style.clipPath = `inset(0 ${100 - Number(el.compareRange.value)}% 0 0)`;
  });
  el.compareRange.addEventListener("input", () => {
    el.before.style.clipPath = `inset(0 ${100 - Number(el.compareRange.value)}% 0 0)`;
  });

  // pointer on the canvas: brush (Fix), crop box (Crop), otherwise pan
  el.canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !current()) return;
    if (event.target.closest(".edit-view-controls")) return; // let its buttons get their clicks (no capture)
    el.canvas.setPointerCapture(event.pointerId);
    if (session.tool === "fix") {
      brush.drawing = true;
      brush.last = null;
      strokeTo(imagePoint(event));
    } else if (session.tool === "crop") {
      const [px, py] = imagePoint(event);
      const { x, y, w, h } = session.crop;
      const inside = px >= x && px <= x + w && py >= y && py <= y + h;
      session.cropDrag = inside ? { mode: "move", dx: px - x, dy: py - y } : { mode: "draw", ox: px, oy: py };
    } else {
      view.panning = { x: event.clientX - view.x, y: event.clientY - view.y };
    }
  });
  el.canvas.addEventListener("pointermove", (event) => {
    if (brush.drawing) strokeTo(imagePoint(event));
    else if (session.cropDrag) {
      const [px, py] = imagePoint(event);
      const drag = session.cropDrag;
      if (drag.mode === "move") {
        session.crop = constrain({ ...session.crop, x: px - drag.dx, y: py - drag.dy });
      } else {
        session.crop = constrain({
          x: Math.min(drag.ox, px), y: Math.min(drag.oy, py), w: Math.abs(px - drag.ox), h: Math.abs(py - drag.oy),
        });
      }
      drawCrop();
    } else if (view.panning) {
      view.x = event.clientX - view.panning.x;
      view.y = event.clientY - view.panning.y;
      applyView();
    }
  });
  for (const type of ["pointerup", "pointercancel"]) {
    el.canvas.addEventListener(type, () => {
      brush.drawing = false;
      brush.last = null;
      session.cropDrag = null;
      view.panning = null;
    });
  }

  el.brush.addEventListener("input", () => { el.brushValue.textContent = el.brush.value; });
  el.paint.addEventListener("click", () => setBrushMode(false));
  el.erase.addEventListener("click", () => setBrushMode(true));
  el.clearMask.addEventListener("click", () => el.mask.getContext("2d").clearRect(0, 0, el.mask.width, el.mask.height));
  el.fixGo.addEventListener("click", fixNow);
  el.restyleStrength.addEventListener("input", () => {
    const value = Number(el.restyleStrength.value);
    el.restyleStrengthValue.textContent = value < 0.4 ? "Subtle" : value < 0.7 ? "Balanced" : "Strong";
  });
  el.restyleGo.addEventListener("click", restyleNow);
  el.upscaleGo.addEventListener("click", upscaleNow);
  for (const chip of el.aspects) {
    chip.addEventListener("click", () => {
      session.aspect = ASPECTS[chip.dataset.aspect];
      for (const other of el.aspects) other.setAttribute("aria-checked", String(other === chip));
      resetCrop();
    });
  }
  el.cropApply.addEventListener("click", cropNow);
  el.rotateLeft.addEventListener("click", () => rotate(-1));
  el.rotateRight.addEventListener("click", () => rotate(1));
  el.flipH.addEventListener("click", () => flip(true));
  el.flipV.addEventListener("click", () => flip(false));
  for (const slider of el.sliders) {
    slider.addEventListener("input", () => {
      session.adjust[slider.dataset.adjust] = Number(slider.value);
      previewLook(session.adjust);
    });
  }
  el.adjustReset.addEventListener("click", resetPreview);
  el.adjustApply.addEventListener("click", () => bakeLook("Adjusted", { ...session.adjust }));
  el.filterApply.addEventListener("click", () => {
    if (!session.filter) return toast("Pick a filter first.", "error");
    const filter = FILTERS[session.filter];
    return bakeLook(filter.label, { ...NEUTRAL, ...filter });
  });
  window.addEventListener("resize", () => { if (!el.workspace.hidden) fitView(); });

  return {
    open,
    openFile,
    shown() {
      renderStyleOptions();
      if (current()) requestAnimationFrame(fitView);
    },
    refresh: renderPanels,
  };
}
