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
  gallery: $("#gallery"), galleryCount: $("#gallery-count"), galleryEmpty: $("#gallery-empty"), loadMore: $("#load-more"),
  status: $("#status"), statusText: $(".status-text"), banner: $("#offline-banner"), toasts: $("#toasts"),
};
const lb = {
  dialog: $("#lightbox"), img: $("#lb-img"), prompt: $("#lb-prompt"), negative: $("#lb-negative"), meta: $("#lb-meta"),
  nometa: $("#lb-nometa"), reuse: $("#lb-reuse"), vary: $("#lb-vary"), download: $("#lb-download"),
  remove: $("#lb-delete"), confirm: $("#lb-confirm"), confirmYes: $("#lb-confirm-yes"), confirmNo: $("#lb-confirm-no"),
  prev: $("#lb-prev"), next: $("#lb-next"), close: $("#lb-close"), upscale: $("#lb-upscale"),
};
const state = {
  images: [], total: 0, busy: false, currentName: null, stageName: null, galleryUnavailable: false,
  models: [], model: null, modelSignature: "", style: "none", count: 1,
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

function syncCount() {
  const model = state.model;
  const tooBig = Math.max(Number(els.width.value), Number(els.height.value)) > MAX_BATCH_SIDE;
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
  const current = state.images.find((image) => image.name === state.stageName);
  current ? showOnStage(current, "Latest") : clearStage();
}

function onGenerated(event) {
  const images = event.images?.length ? event.images : event.image ? [event.image] : [];
  if (!images.length) {
    toast("The image was generated but couldn't be found in ComfyUI's output folder.", "error");
    restoreStage();
    return;
  }
  state.images.unshift(...images);
  state.total += images.length;
  renderGallery();
  const caption = `Seed ${event.seed} · ${event.elapsed}s`;
  if (images.length > 1) showGrid(images, caption);
  else showOnStage(images[0], caption, "from-live");
}

// --- stage -----------------------------------------------------------------

function showGrid(images, caption) {
  state.stageName = images[0].name;
  els.stageGrid.style.setProperty("--aspect", String(images[0].width / images[0].height));
  els.stageGrid.replaceChildren(...images.map((image) => {
    const img = document.createElement("img");
    img.src = imageUrl(image.name);
    img.alt = image.params?.prompt ?? image.name;
    img.addEventListener("click", () => openLightbox(state.images.findIndex((item) => item.name === image.name)));
    return img;
  }));
  els.stageImg.hidden = true;
  els.stageEmpty.hidden = true;
  els.stageGrid.hidden = false;
  els.stageCaption.textContent = `${images.length} variations · ${caption}`;
}

function showOnStage(image, caption, animation = "reveal") {
  els.stageGrid.hidden = true;
  state.stageName = image.name;
  els.stageImg.src = imageUrl(image.name);
  els.stageImg.alt = image.params?.prompt ?? image.name;
  els.stageImg.hidden = false;
  els.stageEmpty.hidden = true;
  els.stageCaption.textContent = caption;
  els.stageImg.classList.remove("reveal", "from-live");
  void els.stageImg.offsetWidth; // restart the animation
  els.stageImg.classList.add(animation);
}

function clearStage() {
  els.stageGrid.hidden = true;
  state.stageName = null;
  els.stageImg.hidden = true;
  els.stageEmpty.hidden = false;
  els.stageCaption.textContent = "";
}

// --- gallery ---------------------------------------------------------------

function tile(image, index) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "tile";
  button.setAttribute("aria-label", image.params?.prompt ?? image.name);
  const img = document.createElement("img");
  img.loading = "lazy";
  img.decoding = "async";
  img.alt = "";
  img.addEventListener("load", () => button.classList.add("loaded"));
  img.addEventListener("error", () => button.classList.add("broken"));
  img.src = imageUrl(image.name, "thumbs");
  button.append(img);
  button.addEventListener("click", () => openLightbox(index));
  return button;
}

function renderGallery() {
  els.gallery.replaceChildren(...state.images.map(tile));
  els.galleryCount.textContent = state.total ? `${state.total} image${state.total === 1 ? "" : "s"}` : "";
  els.galleryEmpty.hidden = state.total > 0;
  if (!state.total) els.galleryEmpty.textContent = "No images yet. Generate your first one above.";
  els.loadMore.hidden = state.images.length >= state.total;
}

async function loadGallery(reset = false) {
  const offset = reset ? 0 : state.images.length;
  try {
    const data = await api(`/api/images?offset=${offset}&limit=${PAGE_SIZE}`);
    state.images = reset ? data.images : state.images.concat(data.images);
    state.total = data.total;
    state.galleryUnavailable = false;
    renderGallery();
    if (reset && !state.stageName && state.images[0]) showOnStage(state.images[0], "Latest");
  } catch (error) {
    state.galleryUnavailable = true;
    els.galleryEmpty.hidden = false;
    els.galleryEmpty.textContent = error.message;
  }
}

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

// The lightbox tracks its image by name: generations and deletes shift positions in state.images.
const currentIndex = () => state.images.findIndex((image) => image.name === state.currentName);
const currentImage = () => state.images[currentIndex()];

function renderLightbox() {
  const index = currentIndex();
  const image = state.images[index];
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
  lb.download.href = `${imageUrl(image.name)}?download=1`;
  lb.prev.disabled = index <= 0;
  lb.next.disabled = index >= state.total - 1;
  lb.confirm.hidden = true;
  lb.remove.hidden = false;
}

function openLightbox(index) {
  state.currentName = state.images[index].name;
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
  if (target < 0 || target >= state.total) return;
  if (target >= state.images.length) await loadGallery();
  if (target < state.images.length) openLightbox(target);
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
  const index = state.images.findIndex((candidate) => candidate.name === image.name);
  if (index < 0) return;
  state.images.splice(index, 1);
  state.total -= 1;
  renderGallery();
  if (state.stageName === image.name) {
    state.images[0] ? showOnStage(state.images[0], "Latest") : clearStage();
  }
  toast("Image deleted.");
  if (!state.images.length) {
    lb.dialog.close();
    return;
  }
  openLightbox(Math.min(index, state.images.length - 1));
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
  if (online && state.galleryUnavailable) loadGallery(true);
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
  const index = state.images.findIndex((image) => image.name === state.stageName);
  if (index >= 0) openLightbox(index);
});

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
loadGallery(true);
checkStatus();
setInterval(checkStatus, STATUS_INTERVAL_MS);
