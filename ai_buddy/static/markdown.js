// Minimal, safe Markdown for coach feedback: headings, lists, bold/italic, inline code, paragraphs.
// Text is HTML-escaped first, so model output can never inject markup.
"use strict";

(function () {
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ESC[c]);

  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*(\S(?:.*?\S)?)\*\*/g, "<strong>$1</strong>")
      .replace(/__(\S(?:.*?\S)?)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*(\S(?:[^*]*?\S)?)\*(?!\w)/g, "$1<em>$2</em>")
      .replace(/(^|[^_\w])_(\S(?:[^_]*?\S)?)_(?!\w)/g, "$1<em>$2</em>");
  }

  function renderMarkdown(src) {
    const out = [];
    let list = null;
    let para = [];
    const flush = () => {
      if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`);
      para = [];
    };
    const close = () => {
      if (list) out.push(`</${list}>`);
      list = null;
    };
    for (const raw of String(src).split("\n")) {
      const line = raw.trimEnd();
      let m;
      if (!line.trim()) { flush(); close(); continue; }
      if ((m = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/))) {
        flush(); close();
        const level = Math.min(Math.max(m[1].length, 2), 4);
        out.push(`<h${level}>${inline(m[2].replace(/#+\s*$/, ""))}</h${level}>`);
      } else if ((m = line.match(/^\s*[-*+•]\s+(.*)$/))) {
        flush();
        if (list !== "ul") { close(); out.push("<ul>"); list = "ul"; }
        out.push(`<li>${inline(m[1])}</li>`);
      } else if ((m = line.match(/^\s*(\d+)[.)]\s+(.*)$/))) {
        flush();
        if (list !== "ol") { close(); out.push(`<ol start="${Number(m[1])}">`); list = "ol"; }
        out.push(`<li>${inline(m[2])}</li>`);
      } else if (list && /^\s{2,}\S/.test(raw)) {
        out[out.length - 1] = out[out.length - 1].replace(/<\/li>$/, ` ${inline(line.trim())}</li>`);
      } else {
        close();
        para.push(line.trim());
      }
    }
    flush();
    close();
    return out.join("");
  }

  // Plain text of one "## Section" (matched by a case-insensitive substring), ready to be spoken.
  function sectionForSpeech(src, title) {
    const lines = String(src).split("\n");
    const keep = [];
    let on = false;
    for (const line of lines) {
      const h = line.match(/^\s*#{1,6}\s+(.*)$/);
      if (h) { on = h[1].toLowerCase().includes(title.toLowerCase()); continue; }
      if (on) keep.push(line);
    }
    const words = ["One", "Two", "Three", "Four", "Five"];
    return keep
      .map((l) => l.replace(/^\s*(\d+)[.)]\s+/, (_, n) => `${words[n - 1] || n}. `).replace(/^\s*[-*+•]\s+/, ""))
      .join("\n")
      .replace(/[*_`#]/g, "")
      .replace(/\n+/g, " ")
      .trim();
  }

  window.renderMarkdown = renderMarkdown;
  window.sectionForSpeech = sectionForSpeech;
})();
