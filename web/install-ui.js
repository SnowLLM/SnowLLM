import { t, onLanguageChange, applyTranslations } from "./i18n.js"
import { latestReleases, supportsTarget } from "./install-releases.js"

  var TORCH_LINE = '<span class="p">$</span>python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ '
    + '"torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision==0.27.0+rocm7.14.0"\n';
  var SNOWLLM_LINE = '<span class="p">$</span>python -m pip install --upgrade ';
  var CURL_LINE = '<span class="p">$</span>curl -fsSL https://snowllm.dev/install.sh | sh';

  var platform = [navigator.userAgentData?.platform, navigator.platform, navigator.userAgent]
    .filter(Boolean).join(" ");
  var choice = { os: /Windows|Win32|Win64/i.test(platform) ? "windows" : "linux",
    channel: "stable", how: "script", gpu: "gfx1151", apu: "ryzen-ai-max" };
  var releases = { stable: null, dev: null };
  var releaseState = "loading";
  var cmdEl = document.getElementById("install-cmd");
  var noteEl = document.getElementById("install-note");
  var channelNoteEl = document.getElementById("channel-note");
  var copyBtn = document.querySelector('[data-copy="panel-install"]');
  var targetEl = document.getElementById("install-target");
  var params = new URLSearchParams(location.search);
  var apus = Array.from(targetEl.options);
  var axes = { channel: ["stable", "dev"], os: ["linux", "windows", "wsl2"], how: ["script", "pip"],
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
    url.searchParams.set("channel", choice.channel);
    url.searchParams.set("os", choice.os);
    url.searchParams.set("how", choice.how);
    url.searchParams.set("target", choice.gpu);
    url.searchParams.set("apu", choice.apu);
    history.replaceState(null, "", url);
  }

  function renderInstall() {
    var channelButtons = Array.from(document.querySelectorAll('.opts[data-axis="channel"] .opt'));
    channelButtons.forEach(function (button) {
      button.hidden = button.dataset.value === "dev" && releaseState === "ready" && !releases.dev;
      button.disabled = button.hidden || !supportsTarget(releases[button.dataset.value], choice.gpu);
    });
    if (releaseState === "ready" && !releases.dev && choice.channel === "dev") choice.channel = "stable";
    var available = channelButtons.find(function (button) { return !button.disabled; });
    if (!channelButtons.some(function (button) {
      return button.dataset.value === choice.channel && !button.disabled;
    }) && available) {
      choice.channel = available.dataset.value;
    }
    var blocked = !supportsTarget(releases[choice.channel], choice.gpu);
    var scriptBtn = document.querySelector('.opts[data-axis="how"] .opt[data-value="script"]');
    var windows = choice.os === "windows";
    var dev = choice.channel === "dev";
    scriptBtn.disabled = windows || blocked;
    if (windows && choice.how === "script") choice.how = "pip";

    ["stable", "dev"].forEach(function (channel) {
      var versionEl = document.getElementById(channel + "-version");
      versionEl.removeAttribute("data-i18n");
      versionEl.textContent = releases[channel] ? "(" + releases[channel] + ")" : t(releaseState === "loading"
        ? "common.loading" : releaseState === "error" ? "install.versionUnavailable" : "install.noRelease");
    });

    document.querySelectorAll(".opts[data-axis]").forEach(function (group) {
      var axis = group.dataset.axis;
      group.querySelectorAll(".opt").forEach(function (b) {
        b.setAttribute("aria-pressed", !b.disabled && choice[axis] === b.dataset.value ? "true" : "false");
      });
    });

    targetEl.value = choice.apu;
    var scriptArgs = [];
    if (dev) scriptArgs.push("--channel dev");
    if (choice.gpu !== "gfx1151") scriptArgs.push("--target " + choice.gpu);
    var curl = CURL_LINE + (scriptArgs.length ? " -s -- " + scriptArgs.join(" ") : "");
    var packageSpec = dev ? "--pre snowllm"
      : releases.stable ? '"snowllm==' + releases.stable + '"' : "snowllm";
    var missing = releaseState === "ready" && !releases[choice.channel];
    cmdEl.innerHTML = missing || blocked ? "" : choice.how === "script" ? curl
      : TORCH_LINE.replace("device-gfx1151", "device-" + choice.gpu) + SNOWLLM_LINE + packageSpec;
    copyBtn.disabled = missing || blocked;
    channelNoteEl.textContent = missing ? t("install.noChannelRelease", { channel: dev ? "Dev" : "Stable" })
      : dev ? "" : t("install.stable");
    if (releaseState === "error") channelNoteEl.textContent += " " + t("install.versionError");
    channelNoteEl.hidden = blocked || !channelNoteEl.textContent;
    noteEl.hidden = choice.os === "linux";
    noteEl.innerHTML = choice.os === "wsl2" ? t("install.wsl")
      : windows ? t("install.windows") : "";
    applyTranslations(noteEl);
  }

  document.querySelectorAll(".opts[data-axis] .opt").forEach(function (button) {
    button.addEventListener("click", function () {
      if (button.disabled || button.hidden) return;
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

async function loadReleases() {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 8000)
  try {
    const response = await fetch("https://pypi.org/pypi/snowllm/json", { signal: controller.signal })
    if (!response.ok) throw new Error("Cannot load PyPI releases")
    releases = latestReleases(await response.json())
    releaseState = "ready"
  } catch {
    releaseState = "error"
  } finally {
    clearTimeout(timeout)
    const previousChannel = choice.channel
    renderInstall()
    if (choice.channel !== previousChannel) saveChoice()
  }
}

loadReleases()
