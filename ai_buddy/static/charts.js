// Progress trends: small multiples (one single-series line chart per metric, each on its own axis),
// sharing one crosshair and one tooltip. The session table on the History page is the table view.
"use strict";

(function () {
  const SVG = "http://www.w3.org/2000/svg";
  const M = { top: 12, right: 40, bottom: 24, left: 34 };
  const PLOT_H = 112;

  function svgEl(tag, attrs = {}) {
    const e = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  function niceMax(v) {
    if (v <= 0) return 1;
    const p = 10 ** Math.floor(Math.log10(v));
    for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }

  function ticks(max, integer) {
    const step = integer ? Math.max(1, Math.ceil(max / 3)) : niceMax(max / 3);
    const out = [];
    for (let t = 0; t <= max + 1e-9; t += step) out.push(Math.round(t * 100) / 100);
    return out;
  }

  const fmtDate = (iso) => new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const fmtWhen = (iso) => new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
  const fmtVal = (v) => (v == null ? "–" : Number.isInteger(v) ? String(v) : v.toFixed(1));

  function renderTrends(container, sessions, practice) {
    container.replaceChildren();
    const metrics = [
      { key: "wpm", title: "Pace", unit: "words / min", note: `comfortable range ${practice.target_wpm_low}–${practice.target_wpm_high}`,
        band: [practice.target_wpm_low, practice.target_wpm_high] },
      { key: "fillers_per_min", title: "Filler words", unit: "per min", note: "lower is better" },
      { key: "long_pauses", title: "Long pauses", unit: `≥ ${practice.long_pause_s} s per session`, note: "lower is better", integer: true },
    ];
    const grid = document.createElement("div");
    grid.className = "trends";
    container.append(grid);
    const tip = document.createElement("div");
    tip.className = "viz-tip";
    tip.hidden = true;
    grid.append(tip);

    const n = sessions.length;
    const charts = [];
    const width = Math.max(220, Math.floor((grid.clientWidth || 820) / 3) - 26);

    for (const m of metrics) {
      const card = document.createElement("figure");
      card.className = "trend";
      card.style.margin = "0";
      const h3 = document.createElement("h3");
      h3.textContent = `${m.title} · ${m.unit}`;
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = m.note;
      card.append(h3, note);

      const vals = sessions.map((s) => s[m.key] ?? 0);
      const yMax = m.integer ? Math.max(3, Math.ceil(niceMax(Math.max(...vals)))) : niceMax(Math.max(...vals, m.band ? m.band[1] * 1.08 : 1));
      const plotW = width - M.left - M.right;
      const x = (i) => M.left + (n === 1 ? plotW / 2 : (i * plotW) / (n - 1));
      const y = (v) => M.top + PLOT_H - (v / yMax) * PLOT_H;
      const H = M.top + PLOT_H + M.bottom;
      const last = vals[n - 1];
      const svg = svgEl("svg", { viewBox: `0 0 ${width} ${H}`, role: "img", tabindex: "0",
        "aria-label": `${m.title} over ${n} practice sessions; latest ${fmtVal(last)} ${m.unit}. Use left and right arrows to inspect sessions.` });

      for (const t of ticks(yMax, m.integer)) {
        svg.append(svgEl("line", { x1: M.left, x2: width - M.right, y1: y(t), y2: y(t),
          stroke: t === 0 ? "var(--axis)" : "var(--hairline)", "stroke-width": 1, "shape-rendering": "crispEdges" }));
        const lbl = svgEl("text", { x: M.left - 6, y: y(t) + 3.5, "text-anchor": "end", class: "axis-label" });
        lbl.textContent = fmtVal(t);
        svg.append(lbl);
      }
      if (m.band) {
        svg.append(svgEl("rect", { x: M.left, width: plotW, y: y(m.band[1]), height: y(m.band[0]) - y(m.band[1]),
          fill: "var(--band)" }));
        const bl = svgEl("text", { x: width - M.right - 4, y: y(m.band[1]) + 11, "text-anchor": "end", class: "band-label" });
        bl.textContent = "target";
        svg.append(bl);
      }
      const dates = n >= 5 ? [0, Math.floor((n - 1) / 2), n - 1] : n > 1 ? [0, n - 1] : [0];
      for (const i of dates) {
        const xl = svgEl("text", { x: x(i), y: H - 6, "text-anchor": n === 1 ? "middle" : i === 0 ? "start" : i === n - 1 ? "end" : "middle",
          class: "axis-label" });
        xl.textContent = fmtDate(sessions[i].started_at);
        svg.append(xl);
      }
      if (n > 1) {
        const d = vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
        svg.append(svgEl("path", { d, fill: "none", stroke: "var(--series-1)", "stroke-width": 2,
          "stroke-linejoin": "round", "stroke-linecap": "round" }));
      }
      if (n <= 16) {
        vals.forEach((v, i) => svg.append(svgEl("circle", { cx: x(i), cy: y(v), r: i === n - 1 ? 4.5 : 4,
          fill: "var(--series-1)", stroke: "var(--surface)", "stroke-width": 2 })));
      } else {
        svg.append(svgEl("circle", { cx: x(n - 1), cy: y(last), r: 4.5, fill: "var(--series-1)", stroke: "var(--surface)", "stroke-width": 2 }));
      }
      const end = svgEl("text", { x: x(n - 1) + 9, y: y(last) + 4, class: "end-label" });
      end.textContent = fmtVal(last);
      svg.append(end);

      const cross = svgEl("line", { y1: M.top, y2: M.top + PLOT_H, stroke: "var(--axis-ink)", "stroke-width": 1,
        visibility: "hidden", "pointer-events": "none" });
      const hot = svgEl("circle", { r: 5.5, fill: "var(--series-1)", stroke: "var(--surface)", "stroke-width": 2,
        visibility: "hidden", "pointer-events": "none" });
      svg.append(cross, hot);
      const hit = svgEl("rect", { x: M.left - 12, y: 0, width: plotW + 24, height: M.top + PLOT_H + 6, fill: "transparent" });
      svg.append(hit);

      card.append(svg);
      grid.insertBefore(card, tip);
      charts.push({ svg, cross, hot, x, y, vals, hit, width });
    }

    let current = -1;
    function show(i, anchorChart) {
      current = i;
      for (const c of charts) {
        const on = i >= 0;
        c.cross.setAttribute("visibility", on ? "visible" : "hidden");
        c.hot.setAttribute("visibility", on ? "visible" : "hidden");
        if (!on) continue;
        c.cross.setAttribute("x1", c.x(i));
        c.cross.setAttribute("x2", c.x(i));
        c.hot.setAttribute("cx", c.x(i));
        c.hot.setAttribute("cy", c.y(c.vals[i]));
      }
      if (i < 0) { tip.hidden = true; return; }
      const s = sessions[i];
      tip.replaceChildren();
      const when = document.createElement("div");
      when.className = "when";
      when.textContent = `${fmtWhen(s.started_at)} · ${s.title}`;
      tip.append(when);
      metrics.forEach((m) => {
        const row = document.createElement("div");
        row.className = "r";
        const key = document.createElement("i");
        key.className = "key";
        const b = document.createElement("b");
        b.textContent = fmtVal(s[m.key]);
        const lab = document.createElement("span");
        lab.textContent = `${m.title.toLowerCase()} (${m.unit})`;
        row.append(key, b, lab);
        tip.append(row);
      });
      tip.hidden = false;
      const c = anchorChart || charts[0];
      const box = c.svg.getBoundingClientRect();
      const gbox = grid.getBoundingClientRect();
      const px = box.left - gbox.left + (c.x(i) / c.width) * box.width;
      const fitsRight = px + 12 + tip.offsetWidth <= gbox.width;
      tip.style.left = `${fitsRight ? px + 12 : Math.max(0, px - tip.offsetWidth - 12)}px`;
      tip.style.top = `${box.top - gbox.top + 8}px`;
    }

    for (const c of charts) {
      c.hit.addEventListener("pointermove", (e) => {
        const box = c.svg.getBoundingClientRect();
        const vx = ((e.clientX - box.left) / box.width) * c.width;
        let best = 0;
        for (let i = 1; i < n; i++) if (Math.abs(c.x(i) - vx) < Math.abs(c.x(best) - vx)) best = i;
        show(best, c);
      });
      c.hit.addEventListener("pointerleave", () => show(-1));
      c.svg.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          const next = current < 0 ? n - 1 : Math.min(n - 1, Math.max(0, current + (e.key === "ArrowRight" ? 1 : -1)));
          show(next, c);
        } else if (e.key === "Escape") show(-1);
      });
      c.svg.addEventListener("blur", () => show(-1));
    }
  }

  window.renderTrends = renderTrends;
})();
