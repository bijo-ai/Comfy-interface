const DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, bad anatomy, text, watermark";
const PAGE_SIZE = 60;
const STATUS_INTERVAL_MS = 10_000;

const $ = (selector) => document.querySelector(selector);
const els = {
  form: $("#composer"), prompt: $("#prompt"), negative: $("#negative"), width: $("#width"), height: $("#height"),
  steps: $("#steps"), cfg: $("#cfg"), seed: $("#seed"), advanced: $("#advanced"),
  generate: $("#generate"), generateLabel: $(".generate-label"), generateFill: $(".generate-fill"),
  shapes: [...document.querySelectorAll(".shape")],
  stageImg: $("#stage-img"), stageEmpty: $("#stage-empty"), stageCaption: $("#stage-caption"),
  gallery: $("#gallery"), galleryCount: $("#gallery-count"), galleryEmpty: $("#gallery-empty"), loadMore: $("#load-more"),
  status: $("#status"), statusText: $(".status-text"), banner: $("#offline-banner"), toasts: $("#toasts"),
};
const lb = {
  dialog: $("#lightbox"), img: $("#lb-img"), prompt: $("#lb-prompt"), negative: $("#lb-negative"), meta: $("#lb-meta"),
  nometa: $("#lb-nometa"), reuse: $("#lb-reuse"), vary: $("#lb-vary"), download: $("#lb-download"),
  remove: $("#lb-delete"), confirm: $("#lb-confirm"), confirmYes: $("#lb-confirm-yes"), confirmNo: $("#lb-confirm-no"),
  prev: $("#lb-prev"), next: $("#lb-next"), close: $("#lb-close"),
};
const state = { images: [], total: 0, busy: false, current: -1, stageName: null, galleryUnavailable: false };

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
  };
}

function fillComposer(params) {
  els.prompt.value = params.prompt;
  els.negative.value = params.negative_prompt;
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

// --- generation ------------------------------------------------------------

function followJob(jobId) {
  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/jobs/${jobId}/events`);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.type === "progress") {
        setProgress(event.step, event.total);
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
  setBusy(true);
  try {
    const { job_id: jobId } = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    onGenerated(await followJob(jobId));
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(false);
  }
}

function onGenerated(event) {
  if (!event.image) {
    toast("The image was generated but couldn't be found in ComfyUI's output folder.", "error");
    return;
  }
  state.images.unshift(event.image);
  state.total += 1;
  renderGallery();
  showOnStage(event.image, `Seed ${event.seed} · ${event.elapsed}s`);
}

// --- stage -----------------------------------------------------------------

function showOnStage(image, caption) {
  state.stageName = image.name;
  els.stageImg.src = imageUrl(image.name);
  els.stageImg.alt = image.params?.prompt ?? image.name;
  els.stageImg.hidden = false;
  els.stageEmpty.hidden = true;
  els.stageCaption.textContent = caption;
  els.stageImg.classList.remove("reveal");
  void els.stageImg.offsetWidth; // restart the animation
  els.stageImg.classList.add("reveal");
}

function clearStage() {
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
  if (p) rows.push(["Seed", p.seed], ["Steps", p.steps], ["CFG", p.cfg]);
  rows.push(["Created", new Date(image.created * 1000).toLocaleString()], ["File", image.name]);
  return rows.flatMap(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    return [dt, dd];
  });
}

function renderLightbox() {
  const image = state.images[state.current];
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
  lb.download.href = `${imageUrl(image.name)}?download=1`;
  lb.prev.disabled = state.current <= 0;
  lb.next.disabled = state.current >= state.total - 1;
  lb.confirm.hidden = true;
  lb.remove.hidden = false;
}

function openLightbox(index) {
  state.current = index;
  renderLightbox();
  if (!lb.dialog.open) lb.dialog.showModal();
}

async function stepLightbox(delta) {
  const target = state.current + delta;
  if (target < 0 || target >= state.total) return;
  if (target >= state.images.length) await loadGallery();
  if (target < state.images.length) openLightbox(target);
}

async function deleteCurrent() {
  const image = state.images[state.current];
  try {
    await api(imageUrl(image.name), { method: "DELETE" });
  } catch (error) {
    toast(error.message, "error");
    return;
  }
  state.images.splice(state.current, 1);
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
  openLightbox(Math.min(state.current, state.images.length - 1));
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
  fillComposer(state.images[state.current].params);
  els.advanced.open = true;
  lb.dialog.close();
  window.scrollTo({ top: 0, behavior: "smooth" });
  els.prompt.focus();
});
lb.vary.addEventListener("click", () => {
  const params = { ...state.images[state.current].params, seed: null };
  fillComposer(params);
  lb.dialog.close();
  window.scrollTo({ top: 0, behavior: "smooth" });
  generate(params);
});
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
loadGallery(true);
checkStatus();
setInterval(checkStatus, STATUS_INTERVAL_MS);
