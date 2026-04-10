(function () {
  const openBtn = document.getElementById("open-info-modal");
  const closeBtn = document.getElementById("close-info-modal");
  const overlay = document.getElementById("info-overlay");
  const modal = document.getElementById("info-modal");

  if (!openBtn || !closeBtn || !overlay || !modal) {
    return;
  }

  function openModal() {
    overlay.hidden = false;
    modal.hidden = false;
    document.body.classList.add("modal-open");
  }

  function closeModal() {
    overlay.hidden = true;
    modal.hidden = true;
    document.body.classList.remove("modal-open");
  }

  openBtn.addEventListener("click", openModal);
  closeBtn.addEventListener("click", closeModal);
  overlay.addEventListener("click", closeModal);

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !modal.hidden) {
      closeModal();
    }
  });

  const nav = performance.getEntriesByType("navigation")[0];
  if (nav && nav.type === "reload" && window.location.search.includes("r=")) {
    window.location.replace(window.location.pathname);
  }

  const formatSelect = document.getElementById("format-select");
  const qualityField = document.getElementById("quality-field");

  if (formatSelect && qualityField) {
    const syncQualityVisibility = function () {
      qualityField.style.display = formatSelect.value === "PNG" ? "none" : "";
    };

    formatSelect.addEventListener("change", syncQualityVisibility);
    syncQualityVisibility();
  }
})();
