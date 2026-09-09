"use strict";

var HEAT_HUE = "#0e8fa6";
var HEAT_INK = "#f3fbfc";

function heatStyle(value, lo, hi) {
  if (value == null || !isFinite(value) || hi <= lo) return {};
  var t = (value - lo) / (hi - lo);
  t = Math.max(0, Math.min(1, t));
  var pct = Math.round(t * 72);
  var style = { background: "color-mix(in oklab, " + HEAT_HUE + " " + pct + "%, transparent)" };
  if (t > 0.55) style.color = HEAT_INK;
  return style;
}

function esc(v) {
  return String(v == null ? "" : v)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function el(tag, attrs, children) {
  var node = document.createElement(tag);
  for (var k in attrs || {}) {
    if (k === "text") node.textContent = attrs[k];
    else if (k === "html") node.innerHTML = attrs[k];
    else node.setAttribute(k, attrs[k]);
  }
  (children || []).forEach(function (c) { if (c) node.appendChild(c); });
  return node;
}

function mountTable(container, rows, columns, opts) {
  opts = opts || {};
  var state = { sort: opts.defaultSort || columns[0].key, dir: opts.defaultDir || "asc",
    filters: {} };

  function visibleRows() {
    var out = rows.filter(function (r) {
      return Object.keys(state.filters).every(function (k) {
        return !state.filters[k] || String(r[k]) === state.filters[k];
      });
    });
    var col = columns.filter(function (c) { return c.key === state.sort; })[0];
    out.sort(function (a, b) {
      var av = col.sortBy ? col.sortBy(a) : a[state.sort];
      var bv = col.sortBy ? col.sortBy(b) : b[state.sort];
      var cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return state.dir === "asc" ? cmp : -cmp;
    });
    return out;
  }

  function render() {
    container.innerHTML = "";
    var visible = visibleRows();

    var filterKeys = columns.filter(function (c) { return c.filter; });
    if (filterKeys.length) {
      var controls = el("div", { class: "controls" });
      filterKeys.forEach(function (c) {
        var values = Array.from(new Set(rows.map(function (r) { return r[c.key]; }))).sort();
        var select = el("select", { "aria-label": "Filter by " + c.label });
        select.appendChild(el("option", { value: "", text: "All " + c.label.toLowerCase() }));
        values.forEach(function (v) {
          select.appendChild(el("option", { value: v, text: v }));
        });
        select.addEventListener("change", function () {
          state.filters[c.key] = select.value;
          render();
        });
        controls.appendChild(el("label", {}, [
          document.createTextNode(c.label + ":"), select,
        ]));
      });
      container.appendChild(controls);
    }

    var heatRange = {};
    columns.forEach(function (c) {
      if (!c.heat) return;
      var vals = visible.map(c.value).filter(function (v) { return v != null && isFinite(v); });
      heatRange[c.key] = [Math.min.apply(null, vals), Math.max.apply(null, vals)];
    });

    if (!visible.length) {
      container.appendChild(el("p", { class: "empty",
        text: "No rows match this filter." }));
      return;
    }

    var wrap = el("div", { class: "scroll" });
    var table = el("table");
    var thead = el("thead");
    var headRow = el("tr");
    columns.forEach(function (c) {
      var th = el("th", { scope: "col" });
      if (c.sortable !== false) {
        th.dataset.key = c.key;
        th.setAttribute("aria-sort", state.sort === c.key
          ? (state.dir === "asc" ? "ascending" : "descending") : "none");
        th.appendChild(document.createTextNode(c.label + " "));
        th.appendChild(el("span", { class: "arrow",
          text: state.sort === c.key ? (state.dir === "asc" ? "↑" : "↓") : "↕" }));
        th.addEventListener("click", function () {
          if (state.sort === c.key) state.dir = state.dir === "asc" ? "desc" : "asc";
          else { state.sort = c.key; state.dir = c.defaultDir || "asc"; }
          render();
        });
      } else {
        th.textContent = c.label;
      }
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = el("tbody");
    visible.forEach(function (r) {
      var tr = el("tr");
      columns.forEach(function (c) {
        var td = el("td", c.className ? { class: c.className(r) } : {});
        td.innerHTML = c.render ? c.render(r) : (r[c.key] == null ? "—" : r[c.key]);
        if (c.heat) {
          var range = heatRange[c.key];
          var style = heatStyle(c.value(r), range[0], range[1]);
          td.dataset.heat = "1";
          for (var prop in style) td.style[prop] = style[prop];
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    container.appendChild(wrap);
  }

  render();
}

var SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs) {
  var node = document.createElementNS(SVG_NS, tag);
  for (var k in attrs || {}) node.setAttribute(k, attrs[k]);
  return node;
}

function niceScale(max) {
  var raw = max / 4 || 1;
  var mag = Math.pow(10, Math.floor(Math.log10(raw)));
  var norm = raw / mag;
  var step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  var top = Math.ceil(max / step) * step;
  var ticks = [];
  for (var v = 0; v <= top + step / 2; v += step) ticks.push(Math.round(v * 100) / 100);
  return { top: top, ticks: ticks };
}

function logScale(lo, hi) {
  var l = Math.floor(Math.log10(lo)), h = Math.ceil(Math.log10(hi));
  var ticks = [];
  for (var k = l; k <= h; k++) ticks.push(Math.pow(10, k));
  return { lo: Math.pow(10, l), top: Math.pow(10, h), ticks: ticks };
}

var CHART_W = 720, CHART_H = 300, CHART_ML = 42, CHART_MR = 12, CHART_MT = 12, CHART_MB = 26;

var METRIC_SPECS = [
  { name: "decode tok/s", section: "Decode", unit: "per second", on: true, unitLabel: "tok/s",
    value: function (p) { return p.decode; },
    num: function (v) { return v.toFixed(1); },
    format: function (v) { return v.toFixed(1) + " tok/s"; },
    note: "Tokens emitted per second \u2014 AL × step/s." },
  { name: "step/s", section: "Decode", unit: "per second", unitLabel: "step/s",
    value: function (p) { return p.step_ms ? 1000 / p.step_ms : null; },
    num: function (v) { return v.toFixed(2); },
    format: function (v) { return v.toFixed(2) + " step/s"; },
    note: "Verify steps per second, on the same axis as decode tok/s: the gap between one "
      + "arm's two lines is its AL." },
  { name: "ms/step", section: "Decode", unit: "ms", unitLabel: "ms",
    value: function (p) { return p.step_ms; },
    num: function (v) { return v.toFixed(1); },
    format: function (v) { return v.toFixed(1) + " ms"; },
    note: "Milliseconds per verify step -- the clean context signal, rising with the KV bytes "
      + "a step reads and nothing else." },
  { name: "AL", section: "Decode", unit: "tok/step", unitLabel: "tok/step",
    value: function (p) { return p.al; },
    num: function (v) { return v.toFixed(2); },
    format: function (v) { return v.toFixed(2) + " tok/step"; },
    note: "Accepted length -- tokens emitted per verify step, drafted plus the guaranteed "
      + "correction token." },
  { name: "accept%", section: "Decode", unit: "accept%", engines: ["llama.cpp"], unitLabel: "%",
    value: function (p) { return p.accept_pct; },
    num: function (v) { return (v * 100).toFixed(1); },
    format: function (v) { return (v * 100).toFixed(1) + "%"; },
    tickFormat: function (v) { return (v * 100).toFixed(0) + "%"; },
    note: "The fraction of offered draft tokens accepted, as llama.cpp reports it. The AL "
      + "above is computed from it." },
  { name: "prefill tok/s", section: "Prefill", unit: "tok/s", on: true, unitLabel: "tok/s",
    value: function (p) { return p.prefill; },
    num: function (v) { return v.toFixed(0); },
    format: function (v) { return v.toFixed(0) + " tok/s"; },
    note: "Prompt tokens per second over the whole prefill." },
  { name: "TTFT s", section: "Prefill", unit: "seconds", log: true, unitLabel: "s",
    value: function (p) { return p.ttft; },
    num: function (v) { return v < 10 ? v.toFixed(2) : v.toFixed(0); },
    format: function (v) { return v < 10 ? v.toFixed(2) + " s" : v.toFixed(0) + " s"; },
    note: "Time to first token, cold, nothing cached. Log axis: half a second to nearly an "
      + "hour across the arms, and a linear one would show only the top rung." },
];

var SERIES_PALETTES = {
  "SnowLLM": ["#0e8fa6", "#1f6feb", "#12a594", "#7c5cd6", "#0f766e", "#3f8f4f", "#2563eb",
              "#0891b2", "#6d28d9", "#047857"],
  "llama.cpp": ["#e07a30", "#c2410c", "#b45309", "#9a3412", "#d97706"],
};

function buildSeriesCatalog(data) {
  var all = data.series || [];
  var order = all.filter(function (s) { return s.featured; })
    .concat(all.filter(function (s) { return !s.featured; }))
    .map(function (s) { return s.key; });
  var seen = {}, colorOf = {};
  order.forEach(function (key) {
    var s = all.filter(function (x) { return x.key === key; })[0];
    var pal = SERIES_PALETTES[s.engine] || SERIES_PALETTES["SnowLLM"];
    var i = seen[s.engine] = (seen[s.engine] == null ? 0 : seen[s.engine] + 1);
    colorOf[key] = pal[i % pal.length];
  });

  return all.map(function (s) {
    return {
      key: s.key, model: s.model, engine: s.engine, arm: s.arm, label: s.label,
      featured: !!s.featured, points: s.points, color: colorOf[s.key],
    };
  });
}

function mountSeriesPicker(container, series, opts) {
  var selSeries = series.filter(function (s) { return s.featured; })
    .map(function (s) { return s.key; });
  if (!selSeries.length) selSeries = series.map(function (s) { return s.key; });

  var phases = [];
  METRIC_SPECS.forEach(function (m) {
    if (phases.indexOf(m.section) === -1) phases.push(m.section);
  });
  var phase = phases[0];
  var selMetrics = {};
  phases.forEach(function (ph) {
    selMetrics[ph] = METRIC_SPECS.filter(function (m) { return m.section === ph && m.on; })
      .map(function (m) { return m.name; });
  });
  var view = "Chart";

  function emit() {
    opts.onChange(selSeries.slice(), METRIC_SPECS.filter(function (m) {
      return m.section === phase && selMetrics[phase].indexOf(m.name) !== -1;
    }), view);
  }

  function chip(text, pressed, onPick) {
    var b = el("button", { class: "opt", type: "button" });
    b.textContent = text;
    b.setAttribute("aria-pressed", pressed ? "true" : "false");
    b.addEventListener("click", function () {
      onPick(b.getAttribute("aria-pressed") !== "true");
    });
    return b;
  }

  function tool(label, node) {
    var box = el("div", { class: "tool" });
    box.appendChild(el("span", { class: "tool-label", text: label }));
    node.setAttribute("aria-label", label);
    if (node.classList.contains("opts")) node.setAttribute("role", "group");
    box.appendChild(node);
    return box;
  }

  function tick(text, checked, color, onToggle) {
    var input = el("input", { type: "checkbox" });
    input.checked = checked;
    var kids = [input];
    if (color) {
      var swatch = el("i", {});
      swatch.style.background = color;
      kids.push(swatch);
    }
    kids.push(document.createTextNode(text));
    var lbl = el("label", { class: "pick" }, kids);
    lbl.classList.toggle("on", checked);
    input.addEventListener("change", function () {
      lbl.classList.toggle("on", input.checked);
      onToggle(input.checked);
    });
    return lbl;
  }

  container.innerHTML = "";
  var bar = el("div", { class: "toolbar" });

  var drop = el("details", { class: "dropdown" });
  var summary = el("summary", {});
  function retitle() {
    summary.textContent = selSeries.length + " of " + series.length + " series";
  }
  drop.appendChild(summary);

  var models = [], byModel = {};
  series.forEach(function (s) {
    if (!byModel[s.model]) { byModel[s.model] = []; models.push(s.model); }
    byModel[s.model].push(s);
  });

  var rowBoxes = [], groupBoxes = [];
  function syncGroups() {
    groupBoxes.forEach(function (g) {
      var on = g.keys.filter(function (k) { return selSeries.indexOf(k) !== -1; }).length;
      g.input.checked = on === g.keys.length;
      g.input.indeterminate = on > 0 && on < g.keys.length;
      g.label.classList.toggle("on", on > 0);
    });
  }

  var list = el("div", { class: "picker stacked" });
  models.forEach(function (model) {
    var keys = byModel[model].map(function (s) { return s.key; });
    var head = tick(model, true, null, function (on) {
      keys.forEach(function (k) {
        var i = selSeries.indexOf(k);
        if (on && i === -1) selSeries.push(k);
        if (!on && i !== -1) selSeries.splice(i, 1);
      });
      rowBoxes.forEach(function (r) {
        if (keys.indexOf(r.key) === -1) return;
        r.input.checked = on;
        r.label.classList.toggle("on", on);
      });
      syncGroups(); retitle(); emit();
    });
    head.classList.add("group-head");
    list.appendChild(head);
    groupBoxes.push({ input: head.querySelector("input"), label: head, keys: keys });

    byModel[model].forEach(function (s) {
      var lbl = tick(s.engine + " · " + s.arm, selSeries.indexOf(s.key) !== -1, s.color,
        function (on) {
          var i = selSeries.indexOf(s.key);
          if (on && i === -1) selSeries.push(s.key);
          if (!on && i !== -1) selSeries.splice(i, 1);
          syncGroups(); retitle(); emit();
        });
      lbl.classList.add("group-item");
      rowBoxes.push({ key: s.key, input: lbl.querySelector("input"), label: lbl });
      list.appendChild(lbl);
    });
  });
  syncGroups();
  retitle();
  drop.appendChild(list);
  bar.appendChild(tool("Series", drop));

  var phaseOpts = el("div", { class: "opts" });
  var phaseChips = [];
  phases.forEach(function (ph) {
    var c = chip(ph, ph === phase, function () {
      phase = ph;
      phaseChips.forEach(function (b) {
        b.setAttribute("aria-pressed", b.textContent === phase ? "true" : "false");
      });
      renderMetrics();
      emit();
    });
    phaseChips.push(c);
    phaseOpts.appendChild(c);
  });
  bar.appendChild(tool("Phase", phaseOpts));

  var metricOpts = el("div", { class: "opts" });
  function renderMetrics() {
    metricOpts.innerHTML = "";
    METRIC_SPECS.filter(function (m) { return m.section === phase; }).forEach(function (m) {
      metricOpts.appendChild(chip(m.name, selMetrics[phase].indexOf(m.name) !== -1,
        function (on) {
          var i = selMetrics[phase].indexOf(m.name);
          if (on && i === -1) selMetrics[phase].push(m.name);
          if (!on && i !== -1) selMetrics[phase].splice(i, 1);
          renderMetrics();
          emit();
        }));
    });
  }
  renderMetrics();
  bar.appendChild(tool("Metric", metricOpts));

  var viewOpts = el("div", { class: "opts" });
  var viewChips = [];
  ["Chart", "Table"].forEach(function (name) {
    var c = chip(name, name === view, function () {
      view = name;
      viewChips.forEach(function (b) {
        b.setAttribute("aria-pressed", b.textContent === view ? "true" : "false");
      });
      emit();
    });
    viewChips.push(c);
    viewOpts.appendChild(c);
  });
  bar.appendChild(tool("View", viewOpts));

  container.appendChild(bar);
  emit();
}

var DASHES = ["", "6,4", "2,3", "9,3,2,3"];

function renderMetricCharts(host, series, selectedKeys, metrics, view) {
  if (!host) return;
  host.innerHTML = "";
  if (!metrics.length) {
    host.appendChild(el("p", { class: "empty", text: "No metric checked." }));
    return;
  }
  var byKey = {};
  series.forEach(function (s) { byKey[s.key] = s; });

  var groups = [], seen = {};
  metrics.forEach(function (m) {
    var id = m.section + " :: " + m.unit;
    if (!seen[id]) { seen[id] = { section: m.section, unit: m.unit, metrics: [] }; groups.push(seen[id]); }
    seen[id].metrics.push(m);
  });

  groups.forEach(function (g) {
    var lines = [], dropped = 0;
    g.metrics.forEach(function (m, mi) {
      selectedKeys.forEach(function (k) {
        var s = byKey[k];
        if (!s) return;
        if (m.engines && m.engines.indexOf(s.engine) === -1) { dropped++; return; }
        lines.push({
          metric: m.name,
          model: s.model,
          engine: s.engine,
          arm: s.arm,
          detail: s.engine + " · " + s.arm + (g.metrics.length > 1 ? " · " + m.name : ""),
          derived: (m.name === "AL" || m.unit === "ms" || m.name === "step/s")
            && s.points.some(function (p) { return p.derived; }),
          label: s.label + (g.metrics.length > 1 ? " · " + m.name : ""),
          color: s.color,
          dash: DASHES[mi % DASHES.length],
          format: m.format,
          num: m.num,
          unitLabel: m.unitLabel,
          points: s.points.map(function (p) {
            return { isl_target: p.isl_target, isl_short: p.isl_short, v: m.value(p) };
          }),
        });
      });
    });

    var group = el("div", { class: "group" });
    group.appendChild(el("h4", { text: g.metrics.map(function (m) { return m.name; }).join(" · ") }));
    var note = g.metrics.map(function (m) { return m.note; }).join(" ");
    if (dropped) note += dropped === 1 ? " One checked series cannot answer it, and draws no line."
      : " " + dropped + " checked series cannot answer it, and draw no line.";
    if (lines.some(function (l) { return l.derived; })) {
      note += " llama.cpp's AL and step time are computed from its accept% and its draft width, "
        + "not measured: a ceiling on AL, a floor on the step.";
    }
    group.appendChild(el("p", { class: "note", text: note }));
    var chart = el("div", { class: "chart" });
    group.appendChild(chart);
    host.appendChild(group);

    if (view === "Table") mountValueTable(chart, lines);
    else mountLineChart(chart, lines, {
      log: g.metrics.some(function (m) { return m.log; }),
      tickFormat: g.metrics[0].tickFormat,
      ariaLabel: g.section + " " + g.unit + " vs input context tokens",
    });
  });
}

function mountValueTable(container, lines) {
  container.innerHTML = "";
  var drawable = lines.filter(function (l) {
    return l.points.some(function (p) { return p.v != null && isFinite(p.v); });
  });
  if (!drawable.length) {
    container.appendChild(el("p", { class: "empty", text: "No data at this selection." }));
    return;
  }

  var rungs = {}, order = [];
  drawable.forEach(function (l) { l.points.forEach(function (p) {
    if (p.v != null && !rungs[p.isl_target]) { rungs[p.isl_target] = p.isl_short; order.push(p.isl_target); }
  }); });
  order.sort(function (a, b) { return a - b; });

  var oneMetric = drawable.every(function (l) { return l.metric === drawable[0].metric; });

  var rows = drawable.map(function (l) {
    var row = { series: l.label, model: l.model, detail: l.detail, color: l.color,
      cell: oneMetric && l.num ? l.num : l.format };
    l.points.forEach(function (p) {
      if (p.v != null && isFinite(p.v)) row["isl" + p.isl_target] = p.v;
    });
    return row;
  });

  var unit = oneMetric ? drawable[0].unitLabel : null;
  var columns = [{ key: "series", label: "Series", sortable: false, className: function () {
      return "series-cell";
    },
    render: function (r) {
      return '<span class="swatch"><i style="background:' + r.color + '"></i>' + r.model
        + '</span><span class="sub">' + r.detail + "</span>";
    } }];
  order.forEach(function (isl) {
    var key = "isl" + isl;
    columns.push({ key: key, label: rungs[isl] + (unit ? " " + unit : ""), heat: oneMetric,
      defaultDir: "desc",
      sortBy: function (r) { return r[key] == null ? -Infinity : r[key]; },
      value: function (r) { return r[key]; },
      render: function (r) { return r[key] == null ? "—" : r.cell(r[key]); } });
  });
  mountTable(container, rows, columns, { defaultSort: "isl" + order[0], defaultDir: "desc" });
}

function mountLineChart(container, lines, opts) {
  opts = opts || {};
  container.innerHTML = "";
  var drawable = lines.filter(function (l) {
    return l.points.some(function (p) { return p.v != null && isFinite(p.v); });
  });
  if (!drawable.length) {
    container.appendChild(el("p", { class: "empty", text: "No data at this selection." }));
    return;
  }

  var rungs = {};
  drawable.forEach(function (l) { l.points.forEach(function (p) {
    if (p.v != null) rungs[p.isl_target] = p.isl_short;
  }); });
  var xs = Object.keys(rungs).map(Number).sort(function (a, b) { return a - b; });

  var values = [];
  drawable.forEach(function (l) { l.points.forEach(function (p) {
    if (p.v != null && isFinite(p.v)) values.push(p.v);
  }); });
  var vmax = Math.max.apply(null, values), vmin = Math.min.apply(null, values);
  var useLog = !!opts.log && vmin > 0 && vmax / vmin > 20;
  var scale = useLog ? logScale(vmin, vmax) : niceScale(vmax);
  var tickFormat = opts.tickFormat || function (v) {
    return v >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : String(v);
  };

  var px0 = CHART_ML, px1 = CHART_W - CHART_MR, py0 = CHART_MT, py1 = CHART_H - CHART_MB;

  function X(isl) {
    var i = xs.indexOf(isl);
    return xs.length === 1 ? (px0 + px1) / 2 : px0 + (i / (xs.length - 1)) * (px1 - px0);
  }
  function Y(v) {
    if (!useLog) return py1 - (v / scale.top) * (py1 - py0);
    var lo = Math.log10(scale.lo), hi = Math.log10(scale.top);
    return py1 - (Math.log10(v) - lo) / (hi - lo) * (py1 - py0);
  }

  var svg = svgEl("svg", { viewBox: "0 0 " + CHART_W + " " + CHART_H, role: "img",
    "aria-label": opts.ariaLabel || "Line chart" });

  scale.ticks.forEach(function (t) {
    var y = Y(t);
    svg.appendChild(svgEl("line", { x1: px0, x2: px1, y1: y, y2: y,
      style: "stroke:var(--hair);stroke-width:1" }));
    var label = svgEl("text", { x: px0 - 8, y: y + 3, "text-anchor": "end",
      style: "font:600 9px var(--mono);fill:var(--faint)" });
    label.textContent = tickFormat(t);
    svg.appendChild(label);
  });
  svg.appendChild(svgEl("line", { x1: px0, x2: px1, y1: py1, y2: py1,
    style: "stroke:var(--line);stroke-width:1" }));

  xs.forEach(function (isl) {
    var label = svgEl("text", { x: X(isl), y: py1 + 18, "text-anchor": "middle",
      style: "font:600 9px var(--mono);fill:var(--faint)" });
    label.textContent = rungs[isl];
    svg.appendChild(label);
  });

  var hoverLine = svgEl("line", { x1: px0, x2: px0, y1: py0, y2: py1,
    style: "stroke:var(--faint);stroke-width:1;stroke-dasharray:2,2;display:none" });

  function valueAt(l, isl) {
    var p = l.points.filter(function (q) { return q.isl_target === isl; })[0];
    return p && p.v != null && isFinite(p.v) ? p.v : null;
  }

  drawable.forEach(function (l) {
    var d = "", drawing = false;
    xs.forEach(function (isl) {
      var v = valueAt(l, isl);
      if (v == null) { drawing = false; return; }
      d += (drawing ? "L" : "M") + X(isl).toFixed(1) + "," + Y(v).toFixed(1) + " ";
      drawing = true;
    });
    svg.appendChild(svgEl("path", { d: d.trim(),
      style: "fill:none;stroke:" + l.color + ";stroke-width:2;stroke-linecap:round;" +
        "stroke-linejoin:round" + (l.dash ? ";stroke-dasharray:" + l.dash : "") }));
    xs.forEach(function (isl) {
      var v = valueAt(l, isl);
      if (v == null) return;
      svg.appendChild(svgEl("circle", { cx: X(isl), cy: Y(v), r: 3, style: "fill:" + l.color }));
    });
  });
  svg.appendChild(hoverLine);

  var labels = svgEl("g", {});
  svg.appendChild(labels);

  var LABEL_GAP = 12;

  function showRung(isl, hovering) {
    hoverLine.setAttribute("x1", X(isl));
    hoverLine.setAttribute("x2", X(isl));
    hoverLine.style.display = hovering ? "" : "none";
    while (labels.firstChild) labels.removeChild(labels.firstChild);

    var here = drawable.map(function (l) { return { line: l, v: valueAt(l, isl) }; })
      .filter(function (d) { return d.v != null; })
      .sort(function (a, b) { return b.v - a.v; });

    var i = xs.indexOf(isl);
    var anchor = i === 0 ? "start" : i === xs.length - 1 ? "end" : "middle";
    var dx = i === 0 ? 7 : i === xs.length - 1 ? -7 : 0;

    var last = null;
    here.forEach(function (d) {
      var y = Y(d.v) - 9;
      if (last != null && y - last < LABEL_GAP) y = last + LABEL_GAP;
      y = Math.min(py1 - 2, Math.max(py0 + 9, y));
      last = y;
      var t = svgEl("text", { x: X(isl) + dx, y: y, "text-anchor": anchor,
        style: "font:600 10px var(--mono);fill:" + d.line.color
          + ";paint-order:stroke;stroke:var(--ground);stroke-width:3px;stroke-linejoin:round" });
      t.textContent = d.line.format(d.v);
      labels.appendChild(t);
    });
  }

  var only = function (get) {
    var v = get(drawable[0]);
    return drawable.every(function (l) { return get(l) === v; }) ? v : null;
  };

  function sharedWords(get) {
    var parts = drawable.map(function (l) { return get(l).split(" "); });
    var out = [];
    for (var i = 0; i < parts[0].length - 1; i++) {
      var w = parts[0][i];
      if (!parts.every(function (p) { return p.length > i + 1 && p[i] === w; })) break;
      out.push(w);
    }
    return out.join(" ");
  }

  var family = drawable.length > 1 ? sharedWords(function (l) { return l.model; }) : "";
  var oneEngine = only(function (l) { return l.engine; });
  var oneArm = only(function (l) { return l.arm; });
  var multi = only(function (l) { return l.metric; }) === null;

  var legend = el("div", { class: "legend chart-legend" });
  drawable.forEach(function (l) {
    var dot = el("i", {});
    dot.style.background = l.color;
    var kids = [dot];
    var name = family ? l.model.slice(family.length).trim() : l.model;
    if (name) kids.push(el("b", { text: name }));
    var rest = [];
    if (!oneEngine) rest.push(l.engine);
    if (!oneArm) rest.push(l.arm);
    if (multi) rest.push(l.metric);
    if (rest.length) kids.push(el("span", { text: rest.join(" \u00b7 ") }));
    legend.appendChild(el("span", { class: "swatch" }, kids));
  });

  var shared = [family, oneEngine, oneArm].filter(Boolean);
  var caption = el("p", { class: "axis-caption",
    text: "Input context, tokens" + (shared.length ? " \u00b7 " + shared.join(" \u00b7 ") : "") });

  function clearHover() { showRung(xs[xs.length - 1], false); }

  xs.forEach(function (isl, i) {
    var xm = X(isl);
    var lo = i === 0 ? px0 : (X(xs[i - 1]) + xm) / 2;
    var hi = i === xs.length - 1 ? px1 : (X(xs[i + 1]) + xm) / 2;
    var hit = svgEl("rect", { x: lo, y: py0, width: Math.max(0, hi - lo), height: py1 - py0,
      style: "fill:transparent", tabindex: "0" });
    hit.addEventListener("mouseenter", function () { showRung(isl, true); });
    hit.addEventListener("focus", function () { showRung(isl, true); });
    hit.addEventListener("mouseleave", clearHover);
    hit.addEventListener("blur", clearHover);
    svg.appendChild(hit);
  });

  container.appendChild(legend);
  container.appendChild(el("div", { class: "chart-svg" }, [svg]));
  container.appendChild(caption);
  clearHover();
}

function loadTable(id, build) {
  var container = document.getElementById(id);
  if (!container) return;
  var src = container.dataset.src;
  container.textContent = "";
  container.appendChild(el("p", { class: "empty", html: container.dataset.empty || "Loading…" }));
  fetch(src, { cache: "no-cache" })
    .then(function (r) {
      if (!r.ok) throw new Error(src + ": " + r.status);
      return r.json();
    })
    .then(function (data) { build(container, data); })
    .catch(function (err) {
      container.innerHTML = "";
      container.appendChild(el("p", { class: "empty",
        text: "Could not load " + src + " (" + err.message + ")." }));
    });
}

loadTable("benchmarks-picker", function (container, data) {
  var series = buildSeriesCatalog(data);
  var host = document.getElementById("benchmarks-charts");
  mountSeriesPicker(container, series, {
    onChange: function (keys, metrics, view) {
      renderMetricCharts(host, series, keys, metrics, view);
    },
  });
});

function humanBytes(n) {
  if (!n) return null;
  var units = ["B", "KiB", "MiB", "GiB", "TiB"];
  var i = 0;
  while (Math.abs(n) >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return (i === 0 ? n.toFixed(0) : n.toFixed(1)) + " " + units[i];
}

function versionParts(v) {
  return String(v || "0").split(".").map(function (x) { return parseInt(x, 10) || 0; });
}

function needsNewer(requires, have) {
  if (!requires) return false;
  var a = versionParts(requires), b = versionParts(have);
  for (var i = 0; i < Math.max(a.length, b.length); i++) {
    var x = a[i] || 0, y = b[i] || 0;
    if (x !== y) return x > y;
  }
  return false;
}

loadTable("recipes-table", function (container, data) {
  var rows = data.recipes;
  var release = data.version;
  var columns = [
    { key: "model", label: "Model", filter: true, className: function () { return "name-cell"; },
      render: function (r) {
        var id = "rec-" + r.id.replace(/[^a-z0-9]+/gi, "-");
        var repos = [{ url: r.url, repo: r.repo }].concat(r.sources || []);
        return '<button class="rowinfo" type="button" popovertarget="' + id + '">'
          + esc(r.model) + "</button>"
          + '<div class="card" popover id="' + id + '">'
          + "<h4>" + esc(r.model) + " &middot; " + esc(r.precision) + "</h4>"
          + (r.summary ? "<p>" + esc(r.summary) + "</p>" : "")
          + "<dl><dt>Source</dt><dd>"
          + repos.map(function (x) {
              return '<a href="' + esc(x.url) + '">' + esc(x.repo) + "</a>";
            }).join("<br>")
          + "</dd><dt>Pull</dt><dd><code>snowllm pull " + esc(r.id) + "</code></dd>"
          + "<dt>Run</dt><dd><code>snowllm " + esc(r.id) + "</code></dd></dl>"
          + "</div>";
      } },
    { key: "precision", label: "Precision", filter: true },
    { key: "spec", label: "Spec", filter: true,
      render: function (r) { return r.spec === "none" ? '<span class="off">none</span>' : r.spec; } },
    { key: "requires", label: "Status", filter: true,
      value: function (r) {
        return needsNewer(r.requires, release) ? "Needs a newer snowllm" : "Supported";
      },
      render: function (r) {
        return needsNewer(r.requires, release)
          ? 'Needs a newer snowllm<span class="sub">snowllm ' + esc(r.requires) + "+</span>"
          : "Supported";
      } },
    { key: "bytes", label: "Size", heat: false, value: function (r) { return r.bytes || 0; },
      render: function (r) { return humanBytes(r.bytes) || "—"; } },
    { key: "id", label: "Pull with", sortable: false,
      render: function (r) { return "<code>snowllm pull " + r.id + "</code>"; } },
  ];
  mountTable(container, rows, columns, { defaultSort: "model" });
});
