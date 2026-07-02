(function () {
  function openTargetDropdown() {
    if (!location.hash) return;

    var id;
    try {
      id = decodeURIComponent(location.hash.slice(1));
    } catch (error) {
      return;
    }
    var target = document.getElementById(id);
    if (!target) return;

    var dropdown = target.closest("details.sd-dropdown");
    if (!dropdown) return;

    dropdown.open = true;
    requestAnimationFrame(function () {
      dropdown.scrollIntoView({ block: "start" });
    });
  }

  window.addEventListener("DOMContentLoaded", openTargetDropdown);
  window.addEventListener("hashchange", openTargetDropdown);
}());
