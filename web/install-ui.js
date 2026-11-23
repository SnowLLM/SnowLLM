import { t, onLanguageChange, applyTranslations } from "./i18n.js"
import { installReleases, supportsTarget } from "./install-releases.js?v=stable-picker-1"

  var TORCH_LINE = '<span class="p">$</span>python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ '
    + '"torch[device-gfx1151]==2.12.0+rocm7.14.0" "torchvision==0.27.0+rocm7.14.0"\n';
  var SNOWLLM_LINE = '<span class="p">$</span>python -m pip install --upgrade ';
  var CURL_LINE = '<span class="p">$</span>curl -fsSL https://snowllm.dev/install.sh | sh';

  var platform = [navigator.userAgentData?.platform, navigator.platform, navigator.userAgent]
    .filter(Boolean).join(" ");
  var choice = { os: /Windows|Win32|Win64/i.test(platform) ? "windows" : "linux",
    channel: "stable", how: "script", gpu: "gfx1151", apu: "ryzen-ai-max" };
  var releases = { stable: [], dev: null };
  var releaseState = "loading";
  var cmdEl = document.getElementById("install-cmd");
  var noteEl = document.getElementById("install-note");
  var channelNoteEl = document.getElementById("channel-note");
  var copyBtn = document.querySelector('[data-copy="panel-install"]');
  var targetEl = document.getElementById("install-target");
  var params = new URLSearchParams(location.search);
  var stableEl = document.getElementById("stable-version");
  choice.version = params.get("version");
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
    if (choice.channel === "stable" && choice.version) url.searchParams.set("version", choice.version);
    else url.searchParams.delete("version");
    history.replaceState(null, "", url);
  }

  function renderInstall() {
    var stable = releases.stable[0]?.version;
    if (releaseState === "ready") {
      for (const option of stableEl.options) option.disabled = !supportsTarget(option.value, choice.gpu);
      if (choice.version && !releases.stable.some(release =>
        release.version === choice.version && supportsTarget(release.version, choice.gpu))) choice.version = null;
      stable = choice.version || stable;
      stableEl.value = stable || "";
    }
    var channelVersions = { stable: stable, dev: releases.dev };
    var channelButtons = Array.from(document.querySelectorAll('.opts[data-axis="channel"] .opt'));
    channelButtons.forEach(function (button) {
      button.hidden = button.dataset.value === "dev" && releaseState === "ready" && !releases.dev;
      button.disabled = button.hidden || !supportsTarget(channelVersions[button.dataset.value], choice.gpu)
        || (releaseState === "ready" && !channelVersions[button.dataset.value]);
    });
    if (releaseState === "ready" && !releases.dev && choice.channel === "dev") choice.channel = "stable";
    var available = channelButtons.find(function (button) { return !button.disabled; });
    if (!channelButtons.some(function (button) {
      return button.dataset.value === choice.channel && !button.disabled;
    }) && available) {
      choice.channel = available.dataset.value;
    }
    var pendingPin = choice.channel === "stable" && choice.version && releaseState !== "ready";
    var blocked = !supportsTarget(channelVersions[choice.channel], choice.gpu);
    var scriptBtn = document.querySelector('.opts[data-axis="how"] .opt[data-value="script"]');
    var windows = choice.os === "windows";
    var dev = choice.channel === "dev";
    scriptBtn.disabled = windows || blocked;
    if (windows && choice.how === "script") choice.how = "pip";

    var versionStatus = t(releaseState === "loading" ? "common.loading"
      : releaseState === "error" ? "install.versionUnavailable" : "install.noRelease");
    document.getElementById("stable-version-row").hidden = dev;
    stableEl.disabled = !stable || blocked;
    if (!releases.stable.length) stableEl.options[0].textContent = versionStatus;
    var devEl = document.getElementById("dev-version");
    devEl.removeAttribute("data-i18n");
    devEl.textContent = releases.dev ? "(" + releases.dev + ")" : versionStatus;

    document.querySelectorAll(".opts[data-axis]").forEach(function (group) {
      var axis = group.dataset.axis;
      group.querySelectorAll(".opt").forEach(function (b) {
        b.setAttribute("aria-pressed", !b.disabled && choice[axis] === b.dataset.value ? "true" : "false");
      });
    });

    targetEl.value = choice.apu;
    var scriptArgs = [];
    if (dev) scriptArgs.push("--channel dev");
    else if (stable) scriptArgs.push("--version " + stable);
    if (choice.gpu !== "gfx1151") scriptArgs.push("--target " + choice.gpu);
    var curl = CURL_LINE + (scriptArgs.length ? " -s -- " + scriptArgs.join(" ") : "");
    var packageSpec = dev ? "--pre snowllm"
      : stable ? '"snowllm==' + stable + '"' : "snowllm";
    var missing = releaseState === "ready" && !channelVersions[choice.channel];
    cmdEl.innerHTML = missing || blocked || pendingPin ? "" : choice.how === "script" ? curl
      : TORCH_LINE.replace("device-gfx1151", "device-" + choice.gpu) + SNOWLLM_LINE + packageSpec;
    copyBtn.disabled = Boolean(missing || blocked || pendingPin);
    channelNoteEl.textContent = missing ? t("install.noChannelRelease", { channel: dev ? "Dev" : "Stable" })
      : releaseState === "error" ? t(pendingPin ? "install.selectedVersionError" : "install.versionError") : "";
    channelNoteEl.hidden = !channelNoteEl.textContent;
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

  stableEl.addEventListener("change", function () {
    choice.version = stableEl.value;
    renderInstall();
    saveChoice();
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
    releases = installReleases(await response.json())
    if (releases.stable.length) {
      stableEl.replaceChildren(...releases.stable.map(({ version, date }) =>
        new Option(date ? version + " · " + date : version, version)))
    }
    releaseState = "ready"
  } catch {
    releaseState = "error"
  } finally {
    clearTimeout(timeout)
    const previousChannel = choice.channel
    const previousVersion = choice.version
    renderInstall()
    if (choice.channel !== previousChannel || choice.version !== previousVersion) saveChoice()
  }
}

loadReleases()
