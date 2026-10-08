const DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, bad anatomy, text, watermark";
const PAGE_SIZE = 60;
const STATUS_INTERVAL_MS = 10_000;

const $ = (selector) => document.querySelector(selector);
const els = {
  form: $("#composer"), prompt: $("#prompt"), negative: $("#negative"), width: $("#width"), height: $("#height"),
  steps: $("#steps"), cfg: $("#cfg"), seed: $("#seed"), advanced: $("#advanced"),
  generate: $("#generate"), generateLabel: $(".generate-label"), generateFill: $(".generate-fill"),
  shapes: [...document.querySelectorAll(".shape[data-shape]")],
  stageImg: $("#stage-img"), stageEmpty: $("#stage-empty"), stageCaption: $("#stage-caption"),
  live: $("#stage-live"), liveCanvas: $("#stage-canvas"), stageGrid: $("#stage-grid"),
  model: $("#model"), modelNote: $("#model-note"), styles: $("#styles"), counts: [...document.querySelectorAll(".count")],
  source: $("#source"), sourcePick: $("#source-pick"), sourceFile: $("#source-file"), sourceThumb: $("#source-thumb"),
  sourceSize: $("#source-size"), sourceClear: $("#source-clear"), strength: $("#strength"), strengthValue: $("#strength-value"),
  sourceFix: $("#source-fix"),
  gallery: $("#gallery"), galleryCount: $("#gallery-count"), galleryEmpty: $("#gallery-empty"), loadMore: $("#load-more"),
  status: $("#status"), statusText: $(".status-text"), banner: $("#offline-banner"), toasts: $("#toasts"),
  createView: $("#create-view"), galleryView: $("#gallery-view"), tabs: [...document.querySelectorAll(".tab")],
  tabCount: $("#tab-count"), search: $("#gallery-search"), filters: [...document.querySelectorAll("#gallery-filters .chip")],
  tabIndicator: $(".tab-indicator"), stage: $(".stage"), stageGlow: $(".stage-glow"),
};
const lb = {
  dialog: $("#lightbox"), img: $("#lb-img"), prompt: $("#lb-prompt"), negative: $("#lb-negative"), meta: $("#lb-meta"),
  nometa: $("#lb-nometa"), reuse: $("#lb-reuse"), vary: $("#lb-vary"), download: $("#lb-download"),
  remove: $("#lb-delete"), confirm: $("#lb-confirm"), confirmYes: $("#lb-confirm-yes"), confirmNo: $("#lb-confirm-no"),
  prev: $("#lb-prev"), next: $("#lb-next"), close: $("#lb-close"), upscale: $("#lb-upscale"), edit: $("#lb-edit"),
  fix: $("#lb-fix"),
};
const painter = {
  dialog: $("#painter"), image: $("#painter-image"), mask: $("#painter-mask"), close: $("#painter-close"),
  paint: $("#mode-paint"), erase: $("#mode-erase"), clear: $("#painter-clear"), brush: $("#brush"),
  brushValue: $("#brush-value"), prompt: $("#painter-prompt"), go: $("#painter-go"),
};
const state = {
  images: [], total: 0, busy: false, currentName: null, stageName: null, galleryUnavailable: false,
  models: [], model: null, modelSignature: "", style: "none", count: 1, source: null, inpaint: false,
  painting: null, // { source fields, width, height } while the painter is open
  stageImages: [], // what the stage shows (one image, or a ×4 grid)
  lbList: [], // the list the lightbox steps through (the filtered gallery, or the stage's images)
  query: "", kind: "", galleryStale: false, tab: "create",
};
const MAX_BATCH_SIDE = 768;
const UPSCALE_MAX_SIDE = 768;

// --- helpers ---------------------------------------------------------------

const imageUrl = (name, kind = "images") => `/api/${kind}/${name.split("/").map(encodeURIComponent).join("/")}`;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const isJson = response.headers.get("content-type")?.includes("application/json");
  const body = isJson ? await response.json() : null;
  if (!response.ok) {
    const error = new Error(body?.error ?? `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return body;
}

function toast(message, kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  els.toasts.append(el);
  setTimeout(() => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 350);
  }, kind === "error" ? 7000 : 3500);
}

// --- composer --------------------------------------------------------------

function syncShapes() {
  for (const button of els.shapes) {
    const match = button.dataset.w === els.width.value && button.dataset.h === els.height.value;
    button.setAttribute("aria-checked", String(match));
  }
  syncCount();
}

// --- models, styles, count --------------------------------------------------

const modelByKey = (key) => state.models.find((model) => model.key === key);
const modelByCkpt = (ckpt) => state.models.find((model) => model.ckpt === ckpt);

// Called at start and on every status tick: picks up ComfyUI coming online and models being added or removed.
async function loadModels() {
  let data;
  try {
    data = await api("/api/models");
  } catch {
    return; // Studio server unreachable; the next status tick retries
  }
  if (!els.styles.children.length) renderStyles(data.styles);
  setInpaintAvailable(Boolean(data.inpaint));
  const available = data.models.filter((model) => model.available);
  const signature = available.map((model) => model.key).join(",");
  if (!available.length || signature === state.modelSignature) return;
  state.models = data.models;
  state.modelSignature = signature;
  els.model.replaceChildren(...available.map((model) => new Option(model.label, model.key)));
  const keep = state.model && available.some((model) => model.key === state.model.key);
  if (keep) selectModel(state.model.key, { keepSize: true });
  else selectModel(data.default ?? available[0].key, { applyDefaults: true });
}

function selectModel(key, { applyDefaults = false, keepSize = false } = {}) {
  const model = modelByKey(key);
  if (!model) return;
  state.model = model;
  els.model.value = key;
  els.modelNote.hidden = !model.heavy;
  els.modelNote.textContent = model.heavy ? "Heavy: about 30–60 s per image on your GPU, 1 at a time, no upscale." : "";
  const shape = els.shapes.find((button) => button.getAttribute("aria-checked") === "true")?.dataset.shape ?? "square";
  for (const button of els.shapes) {
    const [width, height] = model.shapes[button.dataset.shape];
    button.dataset.w = width;
    button.dataset.h = height;
    button.title = `${width}×${height}`;
  }
  if (applyDefaults) {
    els.steps.value = model.steps;
    els.cfg.value = model.cfg;
  }
  if (keepSize) syncShapes();
  else setShape(shape);
  if (state.source) setSource(state.source); // refresh the "made at" size for the new model
}

function setShape(name) {
  const button = els.shapes.find((candidate) => candidate.dataset.shape === name);
  setSize(button.dataset.w, button.dataset.h);
}

function renderStyles(styles) {
  els.styles.replaceChildren(...styles.map((style) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.setAttribute("role", "radio");
    chip.dataset.style = style.key;
    chip.textContent = `${style.emoji} ${style.label}`;
    chip.addEventListener("click", () => setStyle(style.key));
    return chip;
  }));
  setStyle(state.style);
}

function setStyle(key) {
  state.style = key;
  for (const chip of els.styles.children) chip.setAttribute("aria-checked", String(chip.dataset.style === key));
}

function setCount(count) {
  state.count = count;
  for (const button of els.counts) button.setAttribute("aria-checked", String(Number(button.dataset.count) === count));
}

// --- start image (image-to-image) ---------------------------------------------

const IMG2IMG_MAX_SIDE = { sd15: 768, sdxl: 1024 };

function fitSize(width, height, maxSide, minSide = 512) {
  const longSide = Math.max(width, height);
  const scale = Math.min(Math.max(longSide, minSide), maxSide) / longSide;
  return [Math.floor(Math.round(width * scale) / 8) * 8, Math.floor(Math.round(height * scale) / 8) * 8];
}

function sourceSize() {
  if (!state.source) return null;
  return fitSize(state.source.width, state.source.height, IMG2IMG_MAX_SIDE[state.model?.family ?? "sd15"]);
}

function strengthLabel(value) {
  return value < 0.4 ? "Subtle" : value < 0.7 ? "Balanced" : "Strong";
}

function setSource(source) {
  state.source = source;
  els.source.hidden = false;
  els.sourcePick.hidden = true;
  els.sourceThumb.src = source.url;
  const [width, height] = sourceSize();
  els.sourceSize.textContent = `Made at ${width}×${height}`;
  for (const button of els.shapes) button.disabled = true; // the start image sets the shape
  els.sourceFix.hidden = !state.inpaint;
  syncCount();
}

// --- painter (inpainting) -----------------------------------------------------

const MASK_COLOR = "rgb(242, 181, 68)"; // painted solid; the layer itself is see-through (even overlaps)
const brush = { erasing: false, drawing: false, last: null };

function setInpaintAvailable(available) {
  state.inpaint = available;
  els.sourceFix.hidden = !available || !state.source;
}

function openPainter(target) {
  // target: { url, width, height, source_id | source_name, prompt }
  state.painting = target;
  const picture = new Image();
  picture.onload = () => {
    for (const canvas of [painter.image, painter.mask]) {
      canvas.width = picture.naturalWidth;
      canvas.height = picture.naturalHeight;
    }
    painter.image.getContext("2d").drawImage(picture, 0, 0);
    painter.mask.getContext("2d").clearRect(0, 0, painter.mask.width, painter.mask.height);
    fitPainter();
  };
  picture.src = target.url;
  painter.prompt.value = target.prompt ?? "";
  setBrushMode(false);
  if (!painter.dialog.open) painter.dialog.showModal();
}

function fitPainter() {
  // show the picture as large as the editor allows (small 512 px images get scaled up for easier painting)
  const stage = painter.image.closest(".painter-stage");
  const scale = Math.min((stage.clientWidth - 24) / painter.image.width, (stage.clientHeight - 24) / painter.image.height);
  painter.image.style.width = `${Math.floor(painter.image.width * scale)}px`;
  painter.image.style.height = `${Math.floor(painter.image.height * scale)}px`;
}

function setBrushMode(erasing) {
  brush.erasing = erasing;
  painter.paint.setAttribute("aria-checked", String(!erasing));
  painter.erase.setAttribute("aria-checked", String(erasing));
}

function maskPoint(event) {
  const rect = painter.mask.getBoundingClientRect();
  return [
    (event.clientX - rect.left) * (painter.mask.width / rect.width),
    (event.clientY - rect.top) * (painter.mask.height / rect.height),
  ];
}

function strokeTo(point) {
  const ctx = painter.mask.getContext("2d");
  const scale = painter.mask.width / painter.mask.getBoundingClientRect().width;
  ctx.globalCompositeOperation = brush.erasing ? "destination-out" : "source-over";
  ctx.strokeStyle = MASK_COLOR;
  ctx.fillStyle = MASK_COLOR;
  ctx.lineWidth = Number(painter.brush.value) * scale;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  const [x0, y0] = brush.last ?? point;
  ctx.moveTo(x0, y0);
  ctx.lineTo(point[0], point[1]);
  ctx.stroke();
  brush.last = point;
}

function hasPaint() {
  const { data } = painter.mask.getContext("2d").getImageData(0, 0, painter.mask.width, painter.mask.height);
  for (let i = 3; i < data.length; i += 4) if (data[i]) return true;
  return false;
}

function maskBlob() {
  // white where painted, black elsewhere
  const out = document.createElement("canvas");
  out.width = painter.mask.width;
  out.height = painter.mask.height;
  const ctx = out.getContext("2d");
  const source = painter.mask.getContext("2d").getImageData(0, 0, out.width, out.height);
  const pixels = ctx.createImageData(out.width, out.height);
  for (let i = 0; i < source.data.length; i += 4) {
    const value = source.data[i + 3] ? 255 : 0;
    pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = value;
    pixels.data[i + 3] = 255;
  }
  ctx.putImageData(pixels, 0, 0);
  return new Promise((resolve) => out.toBlob(resolve, "image/png"));
}

async function runInpaint() {
  const target = state.painting;
  const prompt = painter.prompt.value.trim();
  if (!prompt) {
    toast("Describe what the picture should show.", "error");
    painter.prompt.focus();
    return;
  }
  if (!hasPaint()) {
    toast("Paint over the area you want to change first.", "error");
    return;
  }
  let maskId;
  try {
    maskId = (await api("/api/sources", { method: "POST", headers: { "Content-Type": "image/png" }, body: await maskBlob() })).id;
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  painter.dialog.close();
  window.scrollTo({ top: 0, behavior: "smooth" });
  const [width, height] = fitSize(target.width, target.height, IMG2IMG_MAX_SIDE.sd15);
  const body = { prompt, mask_id: maskId, style: state.style, negative_prompt: els.negative.value };
  if (target.source_id) body.source_id = target.source_id;
  else body.source_name = target.source_name;
  await runJob("/api/inpaint", body, width, height);
}

function clearSource() {
  state.source = null;
  els.source.hidden = true;
  els.sourcePick.hidden = false;
  els.sourceFile.value = "";
  els.sourceFix.hidden = true;
  for (const button of els.shapes) button.disabled = false;
  syncCount();
}

async function uploadSource(file) {
  if (!file?.type.startsWith("image/")) {
    toast("That file isn't an image.", "error");
    return;
  }
  try {
    const data = await api("/api/sources", { method: "POST", headers: { "Content-Type": file.type }, body: file });
    setSource({ id: data.id, width: data.width, height: data.height, url: `/api/sources/${data.id}` });
    els.prompt.focus();
  } catch (error) {
    toast(error.message, "error");
  }
}

function syncCount() {
  const model = state.model;
  const [checkWidth, checkHeight] = sourceSize() ?? [Number(els.width.value), Number(els.height.value)];
  const tooBig = Math.max(checkWidth, checkHeight) > MAX_BATCH_SIDE;
  const singleOnly = model && model.max_batch < 4;
  const four = els.counts.find((button) => button.dataset.count === "4");
  four.disabled = Boolean(singleOnly || tooBig);
  four.title = singleOnly ? `${model.label} makes 1 image at a time` : tooBig ? `4 at once is limited to ${MAX_BATCH_SIDE}×${MAX_BATCH_SIDE}` : "";
  if (four.disabled && state.count === 4) setCount(1);
}

function setSize(width, height) {
  els.width.value = width;
  els.height.value = height;
  syncShapes();
}

function readParams() {
  const seed = els.seed.value.trim();
  return {
    prompt: els.prompt.value.trim(),
    negative_prompt: els.negative.value,
    width: Number(els.width.value),
    height: Number(els.height.value),
    steps: Number(els.steps.value),
    cfg: Number(els.cfg.value),
    seed: seed === "" ? null : Number(seed),
    model: state.model?.key ?? null,
    style: state.style,
    batch: state.count,
    ...sourceFields(),
  };
}

function sourceFields() {
  if (!state.source) return {};
  const [width, height] = sourceSize(); // the server fits the image the same way; used for the live preview
  return {
    width,
    height,
    strength: Number(els.strength.value),
    ...(state.source.id ? { source_id: state.source.id } : { source_name: state.source.name }),
  };
}

function fillComposer(params) {
  els.prompt.value = params.prompt;
  els.negative.value = params.negative_prompt;
  const model = params.model && modelByCkpt(params.model);
  if (model) selectModel(model.key, { keepSize: true });
  setStyle("none"); // the saved prompt already contains its style keywords
  setSize(params.width, params.height);
  els.steps.value = params.steps;
  els.cfg.value = params.cfg;
  els.seed.value = params.seed ?? "";
}

function setBusy(busy) {
  state.busy = busy;
  els.generate.disabled = busy;
  els.generate.classList.toggle("busy", busy);
  if (busy) {
    setProgress(0, 0);
  } else {
    els.generateLabel.textContent = "Generate";
    els.generateFill.style.transform = "scaleX(0)";
  }
}

function setProgress(step, total) {
  els.generateLabel.textContent = total ? `Step ${step} / ${total}` : "Starting…";
  els.generateFill.style.transform = `scaleX(${total ? step / total : 0})`;
}

// --- live preview ----------------------------------------------------------
// While generating, the stage shows ComfyUI's per-step previews as chunky pixels that get finer each step.

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const LIVE_COLS_START = 14;
const LIVE_COLS_END = 110;
const NOISE_FRAME_MS = 110;
const live = { aspect: 1, noiseTimer: null, latestStep: 0 };

function liveGrid(fraction) {
  const cols = Math.round(LIVE_COLS_START + (LIVE_COLS_END - LIVE_COLS_START) * fraction ** 1.6);
  return [cols, Math.max(1, Math.round(cols / live.aspect))];
}

function drawNoise() {
  const [cols, rows] = liveGrid(0);
  const canvas = els.liveCanvas;
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const pixels = ctx.createImageData(cols, rows);
  for (let i = 0; i < pixels.data.length; i += 4) {
    const v = 14 + Math.random() * 34;
    const warm = Math.random() < 0.06; // occasional amber sparkle
    pixels.data[i] = warm ? 150 + Math.random() * 90 : v;
    pixels.data[i + 1] = warm ? 100 + Math.random() * 60 : v;
    pixels.data[i + 2] = warm ? 30 : v + 6;
    pixels.data[i + 3] = 255;
  }
  ctx.putImageData(pixels, 0, 0);
}

function startLive(width, height) {
  live.aspect = width / height;
  live.latestStep = 0;
  els.live.style.setProperty("--aspect", String(live.aspect));
  els.stageImg.hidden = true;
  els.stageGrid.hidden = true;
  els.stageEmpty.hidden = true;
  els.live.hidden = false;
  els.stageCaption.textContent = "Warming up…";
  drawNoise();
  if (!reducedMotion.matches) live.noiseTimer = setInterval(drawNoise, NOISE_FRAME_MS);
}

function stopNoise() {
  clearInterval(live.noiseTimer);
  live.noiseTimer = null;
}

function stopLive() {
  stopNoise();
  els.live.hidden = true;
}

function showLivePreview(jobId, step, total) {
  live.latestStep = step;
  const frame = new Image();
  frame.onload = () => {
    if (step !== live.latestStep || els.live.hidden) return; // a newer frame is on its way
    stopNoise();
    const [cols, rows] = liveGrid(total ? step / total : 1);
    const canvas = els.liveCanvas;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = true; // average each block, then display it with hard edges
    ctx.drawImage(frame, 0, 0, cols, rows);
  };
  frame.src = `/api/jobs/${jobId}/preview?step=${step}`;
  els.stageCaption.textContent = `Forming · step ${step} / ${total}`;
}

// --- generation ------------------------------------------------------------

function followJob(jobId) {
  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.type === "progress") {
        setProgress(event.step, event.total);
        if (event.preview) showLivePreview(jobId, event.step, event.total);
      } else if (event.type === "done") {
        source.close();
        resolve(event);
      } else if (event.type === "error") {
        source.close();
        reject(new Error(event.message));
      }
    };
    source.onerror = () => {
      source.close();
      reject(new Error("Lost connection to the Studio server."));
    };
  });
}

async function generate(params) {
  if (state.busy) return;
  if (!params.prompt) {
    toast("Write a prompt first.", "error");
    els.prompt.focus();
    return;
  }
  await runJob("/api/generate", params, params.width, params.height);
}

async function upscaleImage(image) {
  await runJob("/api/upscale", { name: image.name }, image.width, image.height);
}

async function runJob(path, body, width, height) {
  if (state.busy) return;
  showTab("create"); // jobs started from the gallery's viewer show their live preview here
  setBusy(true);
  startLive(width, height);
  try {
    const { job_id: jobId } = await api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    onGenerated(await followJob(jobId));
  } catch (error) {
    toast(error.message, "error");
    restoreStage();
  } finally {
    stopLive();
    setBusy(false);
  }
}

function restoreStage() {
  const images = state.stageImages;
  if (images.length > 1) showGrid(images, "Latest");
  else if (images.length === 1) showOnStage(images[0], "Latest");
  else clearStage();
}

function onGenerated(event) {
  const images = event.images?.length ? event.images : event.image ? [event.image] : [];
  if (!images.length) {
    toast("The image was generated but couldn't be found in ComfyUI's output folder.", "error");
    restoreStage();
    return;
  }
  state.galleryStale = true; // the gallery reloads (with its filters) next time it's shown
  refreshCounts();
  const caption = `Seed ${event.seed} · ${event.elapsed}s`;
  if (images.length > 1) showGrid(images, caption);
  else showOnStage(images[0], caption, "from-live");
}

// --- stage -----------------------------------------------------------------

function showGrid(images, caption) {
  state.stageName = images[0].name;
  state.stageImages = images;
  els.stageGrid.style.setProperty("--aspect", String(images[0].width / images[0].height));
  els.stageGrid.replaceChildren(...images.map((image) => {
    const img = document.createElement("img");
    img.src = imageUrl(image.name);
    img.alt = image.params?.prompt ?? image.name;
    img.addEventListener("click", () => openLightbox(image, state.stageImages));
    return img;
  }));
  els.stageImg.hidden = true;
  els.stageEmpty.hidden = true;
  els.stageGrid.hidden = false;
  els.stageCaption.textContent = `${images.length} variations · ${caption}`;
  setStageGlow(images[0]);
}

function setStageGlow(image) {
  // an ambient halo made from the image's own colors (the small thumbnail is plenty once blurred)
  els.stageGlow.classList.remove("on");
  if (!image) return;
  const url = imageUrl(image.name, "thumbs");
  const probe = new Image();
  probe.onload = () => {
    els.stageGlow.style.setProperty("--glow", `url("${url}")`);
    els.stageGlow.classList.add("on");
  };
  probe.src = url;
}

function showOnStage(image, caption, animation = "reveal") {
  els.stageGrid.hidden = true;
  state.stageName = image.name;
  state.stageImages = [image];
  els.stageImg.src = imageUrl(image.name);
  els.stageImg.alt = image.params?.prompt ?? image.name;
  els.stageImg.hidden = false;
  els.stageEmpty.hidden = true;
  els.stageCaption.textContent = caption;
  els.stageImg.classList.remove("reveal", "from-live");
  void els.stageImg.offsetWidth; // restart the animation
  els.stageImg.classList.add(animation);
  setStageGlow(image);
}

function clearStage() {
  els.stageGrid.hidden = true;
  setStageGlow(null);
  state.stageName = null;
  state.stageImages = [];
  els.stageImg.hidden = true;
  els.stageEmpty.hidden = false;
  els.stageCaption.textContent = "";
}

// --- gallery ---------------------------------------------------------------

function tile(image) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "tile";
  button.style.setProperty("--i", String(state.images.indexOf(image) % 24)); // staggered entry
  button.setAttribute("aria-label", image.params?.prompt ?? image.name);
  const img = document.createElement("img");
  img.loading = "lazy";
  img.decoding = "async";
  img.alt = "";
  img.addEventListener("load", () => button.classList.add("loaded"));
  img.addEventListener("error", () => button.classList.add("broken"));
  img.src = imageUrl(image.name, "thumbs");
  button.append(img);
  button.addEventListener("click", () => openLightbox(image, state.images));
  return button;
}

function renderGallery() {
  els.gallery.replaceChildren(...state.images.map(tile));
  const filtered = state.query || state.kind;
  els.galleryCount.textContent = state.total ? `${state.total} image${state.total === 1 ? "" : "s"}${filtered ? " match" : ""}` : "";
  els.galleryEmpty.hidden = state.total > 0;
  if (!state.total) {
    els.galleryEmpty.textContent = filtered
      ? "No images match. Try another search or filter."
      : "No images yet. Make your first one in Create.";
  }
  els.loadMore.hidden = state.images.length >= state.total;
}

async function loadGallery(reset = false) {
  const offset = reset ? 0 : state.images.length;
  const filters = `&q=${encodeURIComponent(state.query)}${state.kind ? `&kind=${state.kind}` : ""}`;
  try {
    const data = await api(`/api/images?offset=${offset}&limit=${PAGE_SIZE}${filters}`);
    state.images = reset ? data.images : state.images.concat(data.images);
    state.total = data.total;
    state.galleryUnavailable = false;
    state.galleryStale = false;
    renderGallery();
  } catch (error) {
    state.galleryUnavailable = true;
    els.galleryEmpty.hidden = false;
    els.galleryEmpty.textContent = error.message;
  }
}

async function refreshCounts() {
  try {
    const counts = await api("/api/images/counts");
    els.tabCount.textContent = counts.all ? String(counts.all) : "";
    for (const badge of document.querySelectorAll(".chip-count")) badge.textContent = counts[badge.dataset.count] || "";
  } catch {
    // offline: counts refresh with the next change
  }
}

async function showLatestOnStage() {
  if (state.stageName) return;
  try {
    const latest = (await api("/api/images?limit=1")).images[0];
    if (latest && !state.stageName) showOnStage(latest, "Latest");
  } catch {
    // ComfyUI's folder unknown yet
  }
}

function setFilter(kind) {
  state.kind = kind;
  for (const chip of els.filters) chip.setAttribute("aria-checked", String(chip.dataset.kind === kind));
  loadGallery(true);
}

let searchTimer = null;
function onSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.query = els.search.value.trim();
    loadGallery(true);
  }, 300);
}

// --- tabs ------------------------------------------------------------------

function showTab(name) {
  state.tab = name === "gallery" ? "gallery" : "create";
  els.createView.hidden = state.tab !== "create";
  els.galleryView.hidden = state.tab !== "gallery";
  for (const tab of els.tabs) {
    if (tab.dataset.tab === state.tab) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
  if (location.hash !== `#${state.tab}`) history.replaceState(null, "", `#${state.tab}`);
  moveTabIndicator();
  if (state.tab === "gallery" && (state.galleryStale || !state.images.length)) loadGallery(true);
}

function moveTabIndicator() {
  const active = els.tabs.find((tab) => tab.dataset.tab === state.tab);
  if (!active) return;
  els.tabIndicator.style.width = `${active.offsetWidth}px`;
  els.tabIndicator.style.transform = `translateX(${active.offsetLeft}px)`;
}

window.addEventListener("resize", moveTabIndicator);
document.fonts?.ready.then(moveTabIndicator); // the web font changes tab widths

// --- lightbox --------------------------------------------------------------

function metaRows(image) {
  const p = image.params;
  const rows = [["Size", image.width ? `${image.width} × ${image.height}` : "Unknown"]];
  if (p) rows.push(["Model", modelByCkpt(p.model)?.label ?? p.model ?? "Unknown"], ["Seed", p.seed], ["Steps", p.steps], ["CFG", p.cfg]);
  rows.push(["Created", new Date(image.created * 1000).toLocaleString()], ["File", image.name]);
  return rows.flatMap(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    return [dt, dd];
  });
}

// The lightbox tracks its image by name (lists shift when images are made or deleted) and steps through
// the list it was opened from: the filtered gallery, or the stage's results.
const currentIndex = () => state.lbList.findIndex((image) => image.name === state.currentName);
const currentImage = () => state.lbList[currentIndex()];
const lightboxTotal = () => (state.lbList === state.images ? state.total : state.lbList.length);

function renderLightbox() {
  const index = currentIndex();
  const image = state.lbList[index];
  const params = image.params;
  lb.img.src = imageUrl(image.name);
  lb.img.alt = params?.prompt ?? image.name;
  lb.prompt.textContent = params?.prompt ?? image.name;
  lb.negative.textContent = params?.negative_prompt || "None";
  lb.meta.replaceChildren(...metaRows(image));
  lb.nometa.hidden = Boolean(params);
  for (const button of [lb.reuse, lb.vary]) {
    button.disabled = !params || (button === lb.vary && state.busy);
    button.title = params ? "" : "Settings unavailable for this image";
  }
  const problem = upscaleProblem(image);
  lb.upscale.disabled = Boolean(problem) || state.busy;
  lb.upscale.title = problem ?? "Make a 2× larger, sharper version";
  lb.fix.hidden = !state.inpaint;
  lb.fix.disabled = state.busy || !image.width;
  lb.download.href = `${imageUrl(image.name)}?download=1`;
  lb.prev.disabled = index <= 0;
  lb.next.disabled = index >= lightboxTotal() - 1;
  lb.confirm.hidden = true;
  lb.remove.hidden = false;
}

function openLightbox(image, list = state.images) {
  state.lbList = list;
  state.currentName = image.name;
  renderLightbox();
  if (!lb.dialog.open) lb.dialog.showModal();
}

function upscaleProblem(image) {
  if (!image.width) return "This image can't be read.";
  if (Math.max(image.width, image.height) > UPSCALE_MAX_SIDE) return "Already large: upscaling it would overload your GPU.";
  const model = image.params?.model && modelByCkpt(image.params.model);
  if (model && !model.upscale) return `${model.label} images can't be upscaled.`;
  return null;
}

async function stepLightbox(delta) {
  const target = currentIndex() + delta;
  if (target < 0 || target >= lightboxTotal()) return;
  if (state.lbList === state.images && target >= state.images.length) {
    await loadGallery();
    state.lbList = state.images; // loading more makes a new list
  }
  if (target < state.lbList.length) openLightbox(state.lbList[target], state.lbList);
}

async function deleteCurrent() {
  const image = currentImage();
  try {
    await api(imageUrl(image.name), { method: "DELETE" });
  } catch (error) {
    if (error.status !== 404) { // 404: already removed outside the app, so drop it here too
      toast(error.message, "error");
      return;
    }
  }
  const index = currentIndex();
  const galleryIndex = state.images.findIndex((candidate) => candidate.name === image.name);
  if (galleryIndex >= 0) {
    state.images.splice(galleryIndex, 1);
    state.total -= 1;
  }
  if (state.lbList !== state.images) state.lbList = state.lbList.filter((candidate) => candidate.name !== image.name);
  renderGallery();
  refreshCounts();
  if (state.stageImages.some((candidate) => candidate.name === image.name)) {
    state.stageImages = state.stageImages.filter((candidate) => candidate.name !== image.name);
    state.stageName = null;
    if (state.stageImages.length) restoreStage();
    else {
      clearStage();
      showLatestOnStage();
    }
  }
  toast("Image deleted.");
  if (!state.lbList.length) {
    lb.dialog.close();
    return;
  }
  openLightbox(state.lbList[Math.min(index, state.lbList.length - 1)], state.lbList);
}

// --- status ----------------------------------------------------------------

async function checkStatus() {
  let online = false;
  try {
    online = (await api("/api/status")).comfyui;
  } catch {
    online = false;
  }
  els.status.dataset.state = online ? "online" : "offline";
  els.statusText.textContent = online ? "ComfyUI connected" : "ComfyUI offline";
  els.banner.hidden = online;
  if (online && state.galleryUnavailable) {
    loadGallery(true);
    refreshCounts();
    showLatestOnStage();
  }
  if (online) loadModels();
}

// --- wiring ----------------------------------------------------------------

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  generate(readParams());
});
els.form.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});
for (const button of els.shapes) button.addEventListener("click", () => setSize(button.dataset.w, button.dataset.h));
els.width.addEventListener("input", syncShapes);
els.height.addEventListener("input", syncShapes);
els.loadMore.addEventListener("click", () => loadGallery());
els.stageImg.addEventListener("click", () => {
  if (state.stageImages.length) openLightbox(state.stageImages[0], state.stageImages);
});
for (const chip of els.filters) chip.addEventListener("click", () => setFilter(chip.dataset.kind));
els.search.addEventListener("input", onSearch);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

lb.close.addEventListener("click", () => lb.dialog.close());
lb.dialog.addEventListener("click", (event) => {
  if (event.target === lb.dialog) lb.dialog.close(); // backdrop click
});
lb.dialog.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") stepLightbox(-1);
  if (event.key === "ArrowRight") stepLightbox(1);
});
lb.prev.addEventListener("click", () => stepLightbox(-1));
lb.next.addEventListener("click", () => stepLightbox(1));
lb.reuse.addEventListener("click", () => {
  fillComposer(currentImage().params);
  els.advanced.open = true;
  lb.dialog.close();
  showTab("create");
  window.scrollTo({ top: 0, behavior: "smooth" });
  els.prompt.focus();
});
lb.vary.addEventListener("click", () => {
  fillComposer({ ...currentImage().params, seed: null });
  setCount(1);
  lb.dialog.close();
  window.scrollTo({ top: 0, behavior: "smooth" });
  generate(readParams());
});
lb.edit.addEventListener("click", () => {
  const image = currentImage();
  setSource({ name: image.name, width: image.width, height: image.height, url: imageUrl(image.name) });
  if (!els.prompt.value.trim() && image.params) els.prompt.value = image.params.prompt;
  lb.dialog.close();
  showTab("create");
  window.scrollTo({ top: 0, behavior: "smooth" });
  els.prompt.focus();
});
lb.fix.addEventListener("click", () => {
  const image = currentImage();
  lb.dialog.close();
  openPainter({
    url: imageUrl(image.name), width: image.width, height: image.height, source_name: image.name,
    prompt: image.params?.prompt ?? "",
  });
});
els.sourceFix.addEventListener("click", () => {
  const source = state.source;
  openPainter({
    url: source.url, width: source.width, height: source.height, prompt: els.prompt.value.trim(),
    ...(source.id ? { source_id: source.id } : { source_name: source.name }),
  });
});
painter.close.addEventListener("click", () => painter.dialog.close());
painter.paint.addEventListener("click", () => setBrushMode(false));
painter.erase.addEventListener("click", () => setBrushMode(true));
painter.clear.addEventListener("click", () => painter.mask.getContext("2d").clearRect(0, 0, painter.mask.width, painter.mask.height));
painter.brush.addEventListener("input", () => { painter.brushValue.textContent = painter.brush.value; });
painter.go.addEventListener("click", runInpaint);
window.addEventListener("resize", () => { if (painter.dialog.open) fitPainter(); });
painter.mask.addEventListener("pointerdown", (event) => {
  painter.mask.setPointerCapture(event.pointerId);
  brush.drawing = true;
  brush.last = null;
  strokeTo(maskPoint(event));
});
painter.mask.addEventListener("pointermove", (event) => {
  if (brush.drawing) strokeTo(maskPoint(event));
});
for (const type of ["pointerup", "pointercancel"]) {
  painter.mask.addEventListener(type, () => {
    brush.drawing = false;
    brush.last = null;
  });
}
els.sourcePick.addEventListener("click", () => els.sourceFile.click());
els.sourceFile.addEventListener("change", () => uploadSource(els.sourceFile.files[0]));
els.sourceClear.addEventListener("click", clearSource);
els.strength.addEventListener("input", () => { els.strengthValue.textContent = strengthLabel(Number(els.strength.value)); });
els.form.addEventListener("dragover", (event) => {
  event.preventDefault();
  els.form.classList.add("dragging");
});
els.form.addEventListener("dragleave", () => els.form.classList.remove("dragging"));
els.form.addEventListener("drop", (event) => {
  event.preventDefault();
  els.form.classList.remove("dragging");
  uploadSource(event.dataTransfer.files[0]);
});
document.addEventListener("paste", (event) => {
  const file = [...event.clipboardData.files].find((item) => item.type.startsWith("image/"));
  if (file) {
    event.preventDefault();
    uploadSource(file);
  }
});
lb.upscale.addEventListener("click", () => {
  const image = currentImage();
  lb.dialog.close();
  window.scrollTo({ top: 0, behavior: "smooth" });
  upscaleImage(image);
});
els.model.addEventListener("change", () => selectModel(els.model.value, { applyDefaults: true }));
for (const button of els.counts) button.addEventListener("click", () => setCount(Number(button.dataset.count)));
lb.remove.addEventListener("click", () => {
  lb.confirm.hidden = false;
  lb.remove.hidden = true;
  lb.confirmNo.focus();
});
lb.confirmNo.addEventListener("click", () => {
  lb.confirm.hidden = true;
  lb.remove.hidden = false;
});
lb.confirmYes.addEventListener("click", deleteCurrent);

els.negative.value = DEFAULT_NEGATIVE;
loadModels();
showTab(location.hash.slice(1));
if (state.tab === "create") loadGallery(true); // keeps the lightbox's gallery list ready
showLatestOnStage();
refreshCounts();
checkStatus();
setInterval(checkStatus, STATUS_INTERVAL_MS);
