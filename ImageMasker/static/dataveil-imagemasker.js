"use strict";

const imageFile = document.querySelector("#image-file");
const imageName = document.querySelector("#image-name");
const detectButton = document.querySelector("#detect-button");
const detectError = document.querySelector("#detect-error");
const maskWorkspace = document.querySelector("#mask-workspace");
const canvas = document.querySelector("#preview-canvas");
const context = canvas.getContext("2d");
const regionList = document.querySelector("#region-list");
const peopleCount = document.querySelector("#people-count");
const drawRegionButton = document.querySelector("#draw-region-button");
const clearRegionsButton = document.querySelector("#clear-regions-button");
const drawHint = document.querySelector("#draw-hint");
const effectSelect = document.querySelector("#effect-select");
const strengthRange = document.querySelector("#strength-range");
const strengthOutput = document.querySelector("#strength-output");
const strengthLabel = document.querySelector("#strength-label");
const colorField = document.querySelector("#color-field");
const effectColor = document.querySelector("#effect-color");
const maskImageButton = document.querySelector("#mask-image-button");
const maskError = document.querySelector("#mask-error");
const imageResult = document.querySelector("#image-result");
const maskedPreview = document.querySelector("#masked-preview");
const editedImageFile = document.querySelector("#edited-image-file");
const restoreImageButton = document.querySelector("#restore-image-button");
const restoreError = document.querySelector("#restore-error");
const restoreResult = document.querySelector("#restore-result");
const restoredPreview = document.querySelector("#restored-preview");
const automaticRecoveryStatus = document.querySelector("#automatic-recovery-status");
const automaticRecoveryTitle = document.querySelector("#automatic-recovery-title");
const automaticRecoveryCopy = document.querySelector("#automatic-recovery-copy");
const clearAutomaticRecoveryButton = document.querySelector("#clear-automatic-recovery");
const recoveryStatusMark = automaticRecoveryStatus.querySelector(".recovery-status-mark");
const RECOVERY_SESSION_KEY = "dataveil-image-recovery-v1";

let sourceImage = null;
let sourceObjectUrl = null;
let detection = null;
let manualRegions = [];
let drawEnabled = false;
let drawStart = null;
let draftRegion = null;
let maskedBlob = null;
let maskedObjectUrl = null;
let recoveryPayload = null;
let restoredBlob = null;
let restoredObjectUrl = null;

function showError(element, message) {
  element.textContent = message;
  element.hidden = false;
}

function hideError(element) {
  element.textContent = "";
  element.hidden = true;
}

function readAutomaticRecovery() {
  try {
    const stored = window.sessionStorage.getItem(RECOVERY_SESSION_KEY);
    if (!stored) return null;
    const parsed = JSON.parse(stored);
    if (parsed?.format !== "dataveil-image-recovery") {
      window.sessionStorage.removeItem(RECOVERY_SESSION_KEY);
      return null;
    }
    return parsed;
  } catch {
    try {
      window.sessionStorage.removeItem(RECOVERY_SESSION_KEY);
    } catch {
      // Storage can be unavailable in hardened browser modes.
    }
    return null;
  }
}

function updateAutomaticRecoveryStatus() {
  const isReady = Boolean(recoveryPayload);
  automaticRecoveryStatus.dataset.state = isReady ? "ready" : "missing";
  recoveryStatusMark.textContent = isReady ? "✓" : "·";
  automaticRecoveryTitle.textContent = isReady
    ? "Recovery data ready"
    : "Waiting for protected image";
  automaticRecoveryCopy.textContent = isReady
    ? "DataVeil will attach it automatically. You only need to upload the GenAI image."
    : "Protect the original image first. DataVeil will keep its recovery data temporarily in this browser tab.";
  clearAutomaticRecoveryButton.hidden = !isReady;
}

function rememberAutomaticRecovery(payload) {
  recoveryPayload = payload;
  try {
    window.sessionStorage.setItem(
      RECOVERY_SESSION_KEY,
      JSON.stringify(payload)
    );
  } catch {
    // Large recovery payloads remain available in memory for this page.
    try {
      window.sessionStorage.removeItem(RECOVERY_SESSION_KEY);
    } catch {
      // Storage can be unavailable in hardened browser modes.
    }
  }
  updateAutomaticRecoveryStatus();
}

function clearAutomaticRecovery() {
  recoveryPayload = null;
  try {
    window.sessionStorage.removeItem(RECOVERY_SESSION_KEY);
  } catch {
    // The in-memory value is still cleared.
  }
  restoreResult.hidden = true;
  updateAutomaticRecoveryStatus();
}

function setBusy(button, busy, busyText) {
  button.disabled = busy;
  button.classList.toggle("is-loading", busy);
  const label = button.querySelector(".button-label");
  if (label) {
    label.textContent = busy ? busyText : "Detect private regions";
  } else {
    button.dataset.originalLabel ||= button.textContent;
    button.textContent = busy ? busyText : button.dataset.originalLabel;
  }
}

function activateWorkflow(view, { scroll = false, updateUrl = true } = {}) {
  const activeView = view === "restore" ? "restore" : "protect";
  document.querySelectorAll(".workflow-tab").forEach((button) => {
    const isActive = button.dataset.view === activeView;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
    button.tabIndex = isActive ? 0 : -1;
  });
  document.querySelector("#protect-view").hidden = activeView !== "protect";
  document.querySelector("#restore-view").hidden = activeView !== "restore";

  if (updateUrl) {
    const url = new URL(window.location.href);
    url.hash = activeView;
    window.history.replaceState(null, "", url);
  }

  if (scroll) {
    document.querySelector(".workflow-chooser").scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }
}

function loadSourceImage(file) {
  if (sourceObjectUrl) URL.revokeObjectURL(sourceObjectUrl);
  sourceObjectUrl = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    sourceImage = image;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    renderCanvas();
  };
  image.src = sourceObjectUrl;
}

function selectedRegionIds() {
  return [...regionList.querySelectorAll('input[type="checkbox"]:checked')]
    .map((checkbox) => checkbox.value);
}

function renderCanvas() {
  if (!sourceImage) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);

  if (detection) {
    const selected = new Set(selectedRegionIds());
    for (const region of detection.regions) {
      const { x, y, width, height } = region.bbox;
      const active = selected.has(region.id);
      context.save();
      context.strokeStyle = active
        ? (region.kind === "face" ? "#f0a34b" : "#36a574")
        : "rgba(255,255,255,.7)";
      context.lineWidth = Math.max(2, canvas.width / 350);
      context.setLineDash(active ? [] : [10, 7]);
      context.strokeRect(x, y, width, height);
      context.fillStyle = active
        ? "rgba(15, 36, 26, .88)"
        : "rgba(50, 58, 54, .72)";
      const labelWidth = Math.min(width, Math.max(84, region.label.length * 7));
      context.fillRect(x, Math.max(0, y - 24), labelWidth, 24);
      context.fillStyle = "#ffffff";
      context.font = `${Math.max(11, canvas.width / 75)}px system-ui`;
      context.fillText(region.label, x + 5, Math.max(16, y - 7));
      context.restore();
    }
  }

  for (const region of [...manualRegions, ...(draftRegion ? [draftRegion] : [])]) {
    context.save();
    context.fillStyle = "rgba(181, 63, 112, .16)";
    context.strokeStyle = "#bd3f75";
    context.lineWidth = Math.max(2, canvas.width / 350);
    context.fillRect(region.x, region.y, region.width, region.height);
    context.strokeRect(region.x, region.y, region.width, region.height);
    context.restore();
  }
}

function renderRegions() {
  regionList.replaceChildren();
  peopleCount.textContent =
    `${detection.person_count} ${detection.person_count === 1 ? "person" : "people"}`;

  if (!detection.regions.length) {
    const empty = document.createElement("p");
    empty.className = "empty-regions";
    empty.textContent =
      "No people were detected. Draw one or more custom boxes on the image.";
    regionList.append(empty);
    return;
  }

  const hasFaces = detection.regions.some((region) => region.kind === "face");
  for (const region of detection.regions) {
    const label = document.createElement("label");
    label.className = `region-choice ${region.kind}`;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = region.id;
    checkbox.checked = hasFaces ? region.kind === "face" : region.kind === "body";
    checkbox.addEventListener("change", renderCanvas);

    const copy = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = region.label;
    const confidence = document.createElement("small");
    confidence.textContent = `${Math.round(region.confidence * 100)}% confidence`;
    copy.append(title, confidence);

    const kind = document.createElement("b");
    kind.textContent = region.kind;
    label.append(checkbox, copy, kind);
    regionList.append(label);
  }
}

async function responseJson(response, fallback) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || fallback);
  return payload;
}

async function detectRegions() {
  hideError(detectError);
  if (!imageFile.files.length) {
    showError(detectError, "Choose a PNG, JPG, or JPEG image first.");
    return;
  }
  setBusy(detectButton, true, "Detecting");
  try {
    const form = new FormData();
    form.append("image", imageFile.files[0]);
    const response = await fetch("/api/image/detect", {
      method: "POST",
      body: form,
      cache: "no-store",
    });
    detection = await responseJson(response, "Image detection failed.");
    manualRegions = [];
    renderRegions();
    renderCanvas();
    maskWorkspace.hidden = false;
    maskWorkspace.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(
      detectError,
      error instanceof Error ? error.message : "Image detection failed."
    );
  } finally {
    setBusy(detectButton, false);
  }
}

function canvasPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: Math.round((event.clientX - bounds.left) * canvas.width / bounds.width),
    y: Math.round((event.clientY - bounds.top) * canvas.height / bounds.height),
  };
}

function normalizeBox(start, end) {
  return {
    id: `manual-${Date.now()}`,
    x: Math.max(0, Math.min(start.x, end.x)),
    y: Math.max(0, Math.min(start.y, end.y)),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
}

function toggleDrawing() {
  drawEnabled = !drawEnabled;
  drawRegionButton.classList.toggle("active", drawEnabled);
  canvas.classList.toggle("drawing", drawEnabled);
  drawHint.textContent = drawEnabled
    ? "Drawing enabled — drag a box over a private area."
    : "Drag on the image to cover anything detection missed.";
}

canvas.addEventListener("pointerdown", (event) => {
  if (!drawEnabled) return;
  event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  drawStart = canvasPoint(event);
});
canvas.addEventListener("pointermove", (event) => {
  if (!drawEnabled || !drawStart) return;
  draftRegion = normalizeBox(drawStart, canvasPoint(event));
  renderCanvas();
});
canvas.addEventListener("pointerup", (event) => {
  if (!drawEnabled || !drawStart) return;
  const region = normalizeBox(drawStart, canvasPoint(event));
  if (region.width >= 5 && region.height >= 5) manualRegions.push(region);
  drawStart = null;
  draftRegion = null;
  renderCanvas();
});

function base64ToBlob(value, mediaType) {
  const bytes = atob(value);
  const array = new Uint8Array(bytes.length);
  for (let index = 0; index < bytes.length; index += 1) {
    array[index] = bytes.charCodeAt(index);
  }
  return new Blob([array], { type: mediaType });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function maskImage() {
  hideError(maskError);
  const selected = selectedRegionIds();
  if (!selected.length && !manualRegions.length) {
    showError(maskError, "Select a detected region or draw a custom box.");
    return;
  }
  setBusy(maskImageButton, true, "Protecting");
  try {
    const form = new FormData();
    form.append("image", imageFile.files[0]);
    form.append("selected_region_ids", JSON.stringify(selected));
    form.append("manual_regions", JSON.stringify(manualRegions));
    form.append("effect", effectSelect.value);
    form.append("strength", strengthRange.value);
    form.append("color", effectColor.value);
    const response = await fetch("/api/image/mask", {
      method: "POST",
      body: form,
      cache: "no-store",
    });
    const payload = await responseJson(response, "Image masking failed.");
    maskedBlob = base64ToBlob(payload.image_base64, payload.media_type);
    rememberAutomaticRecovery(payload.recovery);
    if (maskedObjectUrl) URL.revokeObjectURL(maskedObjectUrl);
    maskedObjectUrl = URL.createObjectURL(maskedBlob);
    maskedPreview.src = maskedObjectUrl;
    imageResult.hidden = false;
    imageResult.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(
      maskError,
      error instanceof Error ? error.message : "Image masking failed."
    );
  } finally {
    setBusy(maskImageButton, false);
  }
}

async function restoreImage() {
  hideError(restoreError);
  if (!editedImageFile.files.length) {
    showError(
      restoreError,
      "Choose the image returned by GenAI."
    );
    return;
  }
  if (!recoveryPayload) {
    showError(
      restoreError,
      "Protect the original image first so DataVeil can prepare recovery data automatically."
    );
    return;
  }
  setBusy(restoreImageButton, true, "Restoring");
  try {
    const form = new FormData();
    form.append("edited_image", editedImageFile.files[0]);
    form.append(
      "recovery_file",
      new Blob(
        [JSON.stringify(recoveryPayload)],
        { type: "application/json" }
      ),
      "dataveil-automatic-recovery.json"
    );
    const response = await fetch("/api/image/restore", {
      method: "POST",
      body: form,
      cache: "no-store",
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || "Image restoration failed.");
    }
    restoredBlob = await response.blob();
    if (restoredObjectUrl) URL.revokeObjectURL(restoredObjectUrl);
    restoredObjectUrl = URL.createObjectURL(restoredBlob);
    restoredPreview.src = restoredObjectUrl;
    document.querySelector("#alignment-status").textContent =
      response.headers.get("X-DataVeil-Alignment") === "applied"
        ? "Feature alignment and soft blending applied"
        : "Size mapping and soft blending applied";
    restoreResult.hidden = false;
    restoreResult.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(
      restoreError,
      error instanceof Error ? error.message : "Image restoration failed."
    );
  } finally {
    setBusy(restoreImageButton, false);
  }
}

const effectLabels = {
  blur: "Blur strength",
  pixelate: "Pixel block size",
  mosaic: "Mosaic intensity",
  black: "Black box opacity",
  solid: "Color opacity",
  noise: "Noise intensity",
  sticker: "Sticker opacity",
  silhouette: "Silhouette opacity",
};

document.querySelectorAll(".workflow-tab").forEach((button) => {
  button.addEventListener("click", () => activateWorkflow(button.dataset.view));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const nextView = button.dataset.view === "protect" ? "restore" : "protect";
    activateWorkflow(nextView);
    document.querySelector(`.workflow-tab[data-view="${nextView}"]`).focus();
  });
});
document.querySelectorAll("[data-switch-workflow]").forEach((button) => {
  button.addEventListener("click", () => {
    activateWorkflow(button.dataset.switchWorkflow, { scroll: true });
  });
});
imageFile.addEventListener("change", () => {
  const file = imageFile.files[0];
  imageName.textContent = file?.name || "No image selected";
  detection = null;
  maskWorkspace.hidden = true;
  imageResult.hidden = true;
  if (file) loadSourceImage(file);
});
detectButton.addEventListener("click", detectRegions);
drawRegionButton.addEventListener("click", toggleDrawing);
clearRegionsButton.addEventListener("click", () => {
  manualRegions = [];
  renderCanvas();
});
document.querySelector("#select-all-button").addEventListener("click", () => {
  regionList.querySelectorAll('input[type="checkbox"]').forEach((item) => {
    item.checked = true;
  });
  renderCanvas();
});
document.querySelector("#select-none-button").addEventListener("click", () => {
  regionList.querySelectorAll('input[type="checkbox"]').forEach((item) => {
    item.checked = false;
  });
  renderCanvas();
});
effectSelect.addEventListener("change", () => {
  strengthLabel.textContent = effectLabels[effectSelect.value];
  colorField.hidden = effectSelect.value !== "solid";
});
strengthRange.addEventListener("input", () => {
  strengthOutput.textContent = `${strengthRange.value}%`;
});
maskImageButton.addEventListener("click", maskImage);
document.querySelector("#download-image-button").addEventListener("click", () => {
  if (maskedBlob) downloadBlob(maskedBlob, "dataveil-masked.png");
});
document.querySelector("#download-recovery-button").addEventListener("click", () => {
  if (!recoveryPayload) return;
  downloadBlob(
    new Blob([JSON.stringify(recoveryPayload)], { type: "application/json" }),
    "dataveil-image-recovery.json"
  );
});
editedImageFile.addEventListener("change", () => {
  const file = editedImageFile.files[0];
  document.querySelector("#edited-image-name").textContent =
    file?.name || "No image selected";
  restoreResult.hidden = true;
  if (!file) return;
  if (!recoveryPayload) {
    showError(
      restoreError,
      "Protect the original image first so DataVeil can prepare recovery data automatically."
    );
    return;
  }
  restoreImage();
});
clearAutomaticRecoveryButton.addEventListener("click", clearAutomaticRecovery);
restoreImageButton.addEventListener("click", restoreImage);
document.querySelector("#download-restored-button").addEventListener("click", () => {
  if (restoredBlob) downloadBlob(restoredBlob, "dataveil-restored.png");
});

activateWorkflow(
  window.location.hash.toLowerCase() === "#restore" ? "restore" : "protect",
  { updateUrl: false }
);
recoveryPayload = readAutomaticRecovery();
updateAutomaticRecoveryStatus();
