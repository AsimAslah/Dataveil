"use strict";

(function createDataVeilUI() {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function scrollTo(element, options = {}) {
    if (!element) return;
    element.scrollIntoView({
      behavior: reducedMotion.matches ? "auto" : "smooth",
      block: options.block || "start",
    });
  }

  function setBusy(button, busy, busyText) {
    if (!button) return;
    if (!button.dataset.defaultLabel) {
      const label = button.querySelector(".button-label");
      button.dataset.defaultLabel = label?.textContent || button.textContent.trim();
    }
    button.disabled = busy;
    button.classList.toggle("is-loading", busy);
    button.setAttribute("aria-busy", String(busy));
    const label = button.querySelector(".button-label");
    if (label) {
      label.textContent = busy ? busyText : button.dataset.defaultLabel;
    } else {
      button.textContent = busy ? busyText : button.dataset.defaultLabel;
    }
  }

  function updateDropZone(input) {
    const zone = input?.closest("[data-drop-zone]");
    if (!zone) return;
    const hasFile = Boolean(input.files?.length);
    zone.classList.toggle("has-file", hasFile);
    const mark = zone.querySelector(".upload-mark, .upload-icon");
    if (mark) mark.textContent = hasFile ? "✓" : "↑";
  }

  function setupDropZone(zone) {
    const input = zone.querySelector('input[type="file"]');
    if (!input) return;
    let dragDepth = 0;

    ["dragenter", "dragover"].forEach((eventName) => {
      zone.addEventListener(eventName, (event) => {
        event.preventDefault();
        if (eventName === "dragenter") dragDepth += 1;
        zone.classList.add("is-dragover");
        if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
      });
    });

    zone.addEventListener("dragleave", () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) zone.classList.remove("is-dragover");
    });

    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      dragDepth = 0;
      zone.classList.remove("is-dragover");
      const files = event.dataTransfer?.files;
      if (!files?.length) return;
      const transfer = new DataTransfer();
      transfer.items.add(files[0]);
      input.files = transfer.files;
      input.dispatchEvent(new Event("change", { bubbles: true }));
    });

    input.addEventListener("change", () => updateDropZone(input));
    updateDropZone(input);
  }

  function setupMotionHeading(heading) {
    function clearShake(event) {
      if (event?.animationName && event.animationName !== "dv-heading-shake") return;
      heading.classList.remove("is-heading-shaking");
    }

    function shakeOnLoad() {
      if (reducedMotion.matches) return;
      heading.classList.remove("is-heading-shaking");
      void heading.offsetWidth;
      heading.classList.add("is-heading-shaking");
    }

    heading.addEventListener("animationend", clearShake);
    heading.addEventListener("animationcancel", clearShake);
    shakeOnLoad();
  }

  function toast(message, type = "success", duration = 3200) {
    const region = document.querySelector("[data-toast-region]");
    if (!region || !message) return;
    const item = document.createElement("div");
    item.className = `toast ${type}`;
    item.setAttribute("role", type === "error" ? "alert" : "status");
    item.textContent = message;
    region.append(item);

    window.setTimeout(() => {
      item.classList.add("is-leaving");
      window.setTimeout(() => item.remove(), reducedMotion.matches ? 0 : 160);
    }, duration);
  }

  function confirmAction({
    title = "Confirm action",
    message = "Are you sure you want to continue?",
    confirmLabel = "Continue",
    destructive = false,
  } = {}) {
    const dialog = document.querySelector("[data-confirm-dialog]");
    if (!dialog || typeof dialog.showModal !== "function") {
      return Promise.resolve(window.confirm(message));
    }

    const titleElement = dialog.querySelector("[data-confirm-title]");
    const messageElement = dialog.querySelector("[data-confirm-message]");
    const cancelButton = dialog.querySelector("[data-confirm-cancel]");
    const confirmButton = dialog.querySelector("[data-confirm-accept]");
    titleElement.textContent = title;
    messageElement.textContent = message;
    confirmButton.textContent = confirmLabel;
    confirmButton.classList.toggle("primary", !destructive);
    confirmButton.classList.toggle("destructive", destructive);

    return new Promise((resolve) => {
      function finish(accepted) {
        dialog.removeEventListener("cancel", onCancel);
        dialog.removeEventListener("keydown", onKeydown);
        cancelButton.removeEventListener("click", onCancelClick);
        confirmButton.removeEventListener("click", onConfirm);
        if (dialog.open) dialog.close();
        resolve(accepted);
      }

      function onCancel(event) {
        event.preventDefault();
        finish(false);
      }

      function onCancelClick() { finish(false); }
      function onConfirm() { finish(true); }
      function onKeydown(event) {
        if (event.key !== "Escape") return;
        event.preventDefault();
        finish(false);
      }

      dialog.addEventListener("cancel", onCancel);
      dialog.addEventListener("keydown", onKeydown);
      cancelButton.addEventListener("click", onCancelClick);
      confirmButton.addEventListener("click", onConfirm);
      dialog.showModal();
      cancelButton.focus();
    });
  }

  function refreshReveal(element) {
    if (!element || element.hidden) return;
    element.removeAttribute("data-reveal");
    void element.offsetWidth;
    element.setAttribute("data-reveal", "");
  }

  document.querySelectorAll("[data-drop-zone]").forEach(setupDropZone);
  document.querySelectorAll(".motion-heading").forEach(setupMotionHeading);

  reducedMotion.addEventListener?.("change", (event) => {
    if (!event.matches) return;
    document.querySelectorAll(".motion-heading").forEach((heading) => {
      heading.classList.remove("is-heading-shaking");
    });
  });

  window.DataVeilUI = {
    confirm: confirmAction,
    refreshReveal,
    scrollTo,
    setBusy,
    toast,
    updateDropZone,
  };
}());
