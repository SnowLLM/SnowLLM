import { t, onLanguageChange } from "./i18n.js"

  var TORCH_LINE = '<span class="p">$</span>python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ '
    + '"torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision==0.27.0+rocm7.14.0"\n';
  var SNOWLLM_LINE = '<span class="p">$</span>python -m pip install snowllm snowllm-kernels';
  var CURL_LINE = '<span class="p">$</span>curl -fsSL https://snowllm.dev/install.sh | sh';

  var choice = { os: "linux", how: "script", gpu: "gfx1151" };
  var cmdEl = document.getElementById("install-cmd");
  var noteEl = document.getElementById("install-note");

  function renderInstall() {
    var scriptBtn = document.querySelector('.opts[data-axis="how"] .opt[data-value="script"]');
    var windows = choice.os === "windows";
    scriptBtn.disabled = windows;
    if (windows && choice.how === "script") choice.how = "pip";

    document.querySelectorAll(".opts[data-axis]").forEach(function (group) {
      var axis = group.dataset.axis;
      group.querySelectorAll(".opt").forEach(function (b) {
        b.setAttribute("aria-pressed", choice[axis] === b.dataset.value ? "true" : "false");
      });
    });

    cmdEl.innerHTML = choice.how === "script" ? CURL_LINE : TORCH_LINE + SNOWLLM_LINE;
    noteEl.hidden = choice.os !== "wsl2";
    noteEl.textContent = choice.os === "wsl2" ? t("install.wsl") : "";
  }

  document.querySelectorAll(".opts[data-axis] .opt").forEach(function (button) {
    button.addEventListener("click", function () {
      if (button.disabled) return;
      choice[button.parentNode.dataset.axis] = button.dataset.value;
      if (cmdEl) renderInstall();
    });
  });

  if (cmdEl) renderInstall();

  document.querySelectorAll(".copy").forEach(function (button) {
    button.addEventListener("click", function () {
      var pre = document.getElementById(button.dataset.copy).querySelector("pre");
      var text = pre.textContent.replace(/^\$/gm, "").trim();
      function done(label) {
        button.textContent = label;
        setTimeout(function () { button.textContent = t("common.copy"); }, 1600);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done(t("common.copied")); }, function () { done(t("common.select")); });
      } else {
        done(t("common.select"));
      }
    });
  });

onLanguageChange(renderInstall)
