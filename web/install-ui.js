import { t, onLanguageChange } from "./i18n.js"

  var TORCH_LINE = '<span class="p">$</span>python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ '
    + '"torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision==0.27.0+rocm7.14.0"\n';
  var SNOWLLM_LINE = '<span class="p">$</span>python -m pip install snowllm snowllm-kernels';
  var CURL_LINE = '<span class="p">$</span>curl -fsSL https://snowllm.dev/install.sh | sh';

  var choice = { os: "linux", how: "script", gpu: "gfx1151", apu: "ryzen-ai-max" };
  var cmdEl = document.getElementById("install-cmd");
  var noteEl = document.getElementById("install-note");
  var targetEl = document.getElementById("install-target");
  var params = new URLSearchParams(location.search);
  var apus = Array.from(targetEl.options);
  var axes = { os: ["linux", "wsl2", "windows"], how: ["script", "pip"],
    gpu: apus.map(function (option) { return option.dataset.target; }) };
  Object.keys(axes).forEach(function (axis) {
    var value = params.get(axis === "gpu" ? "target" : axis);
    if (axes[axis].includes(value)) choice[axis] = value;
  });
  var initialApu = apus.find(function (option) { return option.value === params.get("apu"); })
    || apus.find(function (option) { return option.value === choice.apu && option.dataset.target === choice.gpu; })
    || apus.find(function (option) { return option.dataset.target === choice.gpu; });
  choice.apu = initialApu.value;
  choice.gpu = initialApu.dataset.target;

  function saveChoice() {
    var url = new URL(location.href);
    url.searchParams.set("os", choice.os);
    url.searchParams.set("how", choice.how);
    url.searchParams.set("target", choice.gpu);
    url.searchParams.set("apu", choice.apu);
    history.replaceState(null, "", url);
  }

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

    targetEl.value = choice.apu;
    var curl = choice.gpu === "gfx1151" ? CURL_LINE
      : CURL_LINE + " -s -- --target " + choice.gpu;
    cmdEl.innerHTML = choice.how === "script" ? curl
      : TORCH_LINE.replace("device-gfx1151", "device-" + choice.gpu) + SNOWLLM_LINE;
    noteEl.hidden = choice.os !== "wsl2";
    noteEl.textContent = choice.os === "wsl2" ? t("install.wsl") : "";
  }

  document.querySelectorAll(".opts[data-axis] .opt").forEach(function (button) {
    button.addEventListener("click", function () {
      if (button.disabled) return;
      choice[button.parentNode.dataset.axis] = button.dataset.value;
      if (cmdEl) renderInstall();
      saveChoice();
    });
  });

  targetEl.addEventListener("change", function () {
    choice.apu = targetEl.value;
    choice.gpu = targetEl.selectedOptions[0].dataset.target;
    renderInstall();
    saveChoice();
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
