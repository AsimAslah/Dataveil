"use strict";

const sourceText = document.querySelector("#source-text");
const documentFile = document.querySelector("#document-file");
const documentName = document.querySelector("#document-name");
const charCount = document.querySelector("#char-count");
const scanButton = document.querySelector("#scan-button");
const clearButton = document.querySelector("#clear-button");
const errorMessage = document.querySelector("#error-message");
const reviewSection = document.querySelector("#review-section");
const entityList = document.querySelector("#entity-list");
const detectionTotal = document.querySelector("#detection-total");
const maskingType = document.querySelector("#masking-type");
const styleNote = document.querySelector("#style-note");
const maskButton = document.querySelector("#mask-button");
const resultSection = document.querySelector("#result-section");
const maskedText = document.querySelector("#masked-text");
const maskedTotal = document.querySelector("#masked-total");
const statusMessage = document.querySelector("#status-message");
const restoreSource = document.querySelector("#restore-source");
const mappingFile = document.querySelector("#mapping-file");
const mappingName = document.querySelector("#mapping-name");
const restoreError = document.querySelector("#restore-error");
const restoreResultSection = document.querySelector("#restore-result-section");
const restoredText = document.querySelector("#restored-text");

const maximumLength = Number(sourceText.maxLength);
let inputMode = "text";
let detectedText = "";
let detections = [];
let latestResult = null;

function showMessage(element, message) {
  element.textContent = message;
  element.hidden = false;
}

function hideMessage(element) {
  element.textContent = "";
  element.hidden = true;
}

function setBusy(button, busy, busyLabel = "Working") {
  button.disabled = busy;
  button.classList.toggle("is-loading", busy);
  const label = button.querySelector(".button-label");
  if (label) {
    label.textContent = busy ? busyLabel : "Scan private data";
  }
}

function updateCharacterCount() {
  charCount.textContent =
    `${sourceText.value.length.toLocaleString()} / ${maximumLength.toLocaleString()}`;
}

function activateInputMode(mode) {
  inputMode = mode;
  document.querySelectorAll(".input-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.input === mode);
  });
  document.querySelector("#text-input-pane").hidden = mode !== "text";
  document.querySelector("#file-input-pane").hidden = mode !== "file";
  charCount.hidden = mode !== "text";
}

function activateWorkflow(view) {
  document.querySelectorAll(".workflow-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelector("#protect-view").hidden = view !== "protect";
  document.querySelector("#restore-view").hidden = view !== "restore";
}

function entityTypeLabel(value) {
  return value.replaceAll("_", " ").toLowerCase().replace(
    /(^|\s)\S/g,
    (letter) => letter.toUpperCase()
  );
}

function renderDetections() {
  entityList.replaceChildren();
  detectionTotal.textContent = `${detections.length} found`;

  if (!detections.length) {
    const empty = document.createElement("p");
    empty.className = "empty-review";
    empty.textContent = "No matching private details were detected. You can still review the extracted text.";
    entityList.append(empty);
    return;
  }

  for (const entity of detections) {
    const label = document.createElement("label");
    label.className = "entity-choice";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = entity.id;
    checkbox.checked = true;

    const content = document.createElement("span");
    content.className = "entity-choice-content";

    const top = document.createElement("span");
    top.className = "entity-choice-top";
    const type = document.createElement("strong");
    type.textContent = entityTypeLabel(entity.entity_type);
    const confidence = document.createElement("small");
    confidence.textContent = `${Math.round(entity.confidence_score * 100)}%`;
    top.append(type, confidence);

    const original = document.createElement("span");
    original.className = "entity-original";
    original.textContent = entity.original_value;

    content.append(top, original);
    label.append(checkbox, content);
    entityList.append(label);
  }
}

async function parseResponse(response, fallback) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || fallback);
  }
  return payload;
}

async function scanContent() {
  hideMessage(errorMessage);
  setBusy(scanButton, true, "Scanning");
  reviewSection.hidden = true;
  resultSection.hidden = true;

  try {
    let response;
    if (inputMode === "file") {
      if (!documentFile.files.length) {
        throw new Error("Choose a PDF, DOCX, or PPTX document first.");
      }
      const form = new FormData();
      form.append("document", documentFile.files[0]);
      response = await fetch("/api/doc/detect-file", {
        method: "POST",
        body: form,
        cache: "no-store",
      });
    } else {
      response = await fetch("/api/doc/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sourceText.value }),
        cache: "no-store",
      });
    }

    const payload = await parseResponse(
      response,
      "DocMasker could not scan this content."
    );
    detectedText = payload.text;
    detections = payload.detected_entities;
    renderDetections();
    reviewSection.hidden = false;
    reviewSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showMessage(
      errorMessage,
      error instanceof Error ? error.message : "DocMasker could not scan this content."
    );
  } finally {
    setBusy(scanButton, false);
  }
}

function selectedEntityIds() {
  return [...entityList.querySelectorAll('input[type="checkbox"]:checked')]
    .map((checkbox) => checkbox.value);
}

async function maskSelections() {
  hideMessage(errorMessage);
  maskButton.disabled = true;
  try {
    const response = await fetch("/api/doc/mask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: detectedText,
        selected_entity_ids: selectedEntityIds(),
        masking_type: maskingType.value,
      }),
      cache: "no-store",
    });
    latestResult = await parseResponse(
      response,
      "DocMasker could not mask the selected details."
    );
    maskedText.value = latestResult.masked_text;
    maskedTotal.textContent = `${latestResult.detection_count} masked`;
    resultSection.hidden = false;
    showMessage(statusMessage, "Masking completed locally.");
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showMessage(
      errorMessage,
      error instanceof Error ? error.message : "Masking failed."
    );
  } finally {
    maskButton.disabled = false;
  }
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

function downloadText(value, filename, type = "text/plain;charset=utf-8") {
  downloadBlob(new Blob([value], { type }), filename);
}

function downloadRecovery() {
  if (!latestResult) return;
  const recovery = {
    format: "dataveil-text-recovery",
    version: 1,
    masking_type: maskingType.value,
    masked_text: latestResult.masked_text,
    placeholder_mapping: latestResult.placeholder_mapping,
  };
  downloadText(
    JSON.stringify(recovery, null, 2),
    "dataveil-text-recovery.json",
    "application/json"
  );
}

async function restorePrivateText() {
  hideMessage(restoreError);
  if (!restoreSource.value.trim()) {
    showMessage(restoreError, "Paste the AI-edited text first.");
    return;
  }
  if (!mappingFile.files.length) {
    showMessage(restoreError, "Choose the DataVeil recovery file.");
    return;
  }

  try {
    const recovery = JSON.parse(await mappingFile.files[0].text());
    if (
      recovery.format !== "dataveil-text-recovery" ||
      !recovery.placeholder_mapping
    ) {
      throw new Error("This is not a valid DataVeil text recovery file.");
    }
    const response = await fetch("/api/doc/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        masked_text: restoreSource.value,
        placeholder_mapping: recovery.placeholder_mapping,
      }),
      cache: "no-store",
    });
    const payload = await parseResponse(response, "Text restoration failed.");
    restoredText.value = payload.restored_text;
    restoreResultSection.hidden = false;
    restoreResultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showMessage(
      restoreError,
      error instanceof Error ? error.message : "Text restoration failed."
    );
  }
}

function clearWorkspace() {
  sourceText.value = "";
  documentFile.value = "";
  documentName.textContent = "No document selected";
  detectedText = "";
  detections = [];
  latestResult = null;
  reviewSection.hidden = true;
  resultSection.hidden = true;
  entityList.replaceChildren();
  maskedText.value = "";
  hideMessage(errorMessage);
  hideMessage(statusMessage);
  updateCharacterCount();
}

const styleNotes = {
  placeholder: "Best for reliable recovery after AI editing. Keep tokens unchanged.",
  redact: "Typed redaction tokens remain recoverable while their labels are preserved.",
  pseudonym: "Readable aliases are convenient, but AI tools may rewrite them.",
  generalize: "Categories hide detail but can be reordered, making recovery less reliable.",
  remove: "Removed values cannot be restored after the text structure changes.",
};

document.querySelectorAll(".input-tab").forEach((button) => {
  button.addEventListener("click", () => activateInputMode(button.dataset.input));
});
document.querySelectorAll(".workflow-tab").forEach((button) => {
  button.addEventListener("click", () => activateWorkflow(button.dataset.view));
});
sourceText.addEventListener("input", updateCharacterCount);
documentFile.addEventListener("change", () => {
  documentName.textContent = documentFile.files[0]?.name || "No document selected";
});
mappingFile.addEventListener("change", () => {
  mappingName.textContent = mappingFile.files[0]?.name || "No recovery file selected";
});
maskingType.addEventListener("change", () => {
  styleNote.textContent = styleNotes[maskingType.value];
});
scanButton.addEventListener("click", scanContent);
maskButton.addEventListener("click", maskSelections);
clearButton.addEventListener("click", clearWorkspace);
document.querySelector("#select-all-button").addEventListener("click", () => {
  entityList.querySelectorAll('input[type="checkbox"]').forEach((item) => {
    item.checked = true;
  });
});
document.querySelector("#select-none-button").addEventListener("click", () => {
  entityList.querySelectorAll('input[type="checkbox"]').forEach((item) => {
    item.checked = false;
  });
});
document.querySelector("#copy-button").addEventListener("click", async () => {
  await navigator.clipboard.writeText(maskedText.value);
  showMessage(statusMessage, "Protected text copied.");
});
document.querySelector("#download-text-button").addEventListener("click", () => {
  downloadText(maskedText.value, "dataveil-masked-text.txt");
});
document.querySelector("#download-mapping-button").addEventListener("click", downloadRecovery);
document.querySelector("#restore-button").addEventListener("click", restorePrivateText);
document.querySelector("#download-restored-button").addEventListener("click", () => {
  downloadText(restoredText.value, "dataveil-restored-text.txt");
});

updateCharacterCount();
