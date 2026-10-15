// AI Buddy front end: microphone capture, streaming playback, the three modes, tools and settings.
"use strict";

const $ = (id) => document.getElementById(id);
const EMOJI = /[\p{Extended_Pictographic}\u{1F1E6}-\u{1F1FF}\u{FE0F}\u{200D}\u{20E3}]/gu;
const stripEmoji = (s) => s.replace(EMOJI, "").replace(/[ \t]{2,}/g, " ");
const fmtClock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const fmtWhen = (iso) => new Date(iso).toLocaleString(undefined, { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------- state

const state = {
  config: null,
  mode: "chat",            // the visible tab
  serverMode: null,        // the mode last started on the server
  connected: false,
  warm: false,
  models: null,
  server: "idle",
  detail: "",
  recording: false,
  talkIntent: false,
  playing: false,
  lastPlaybackEnd: 0,
  handsFree: false,
  replies: new Map(),      // reply id -> {msg, text, readout}
  pendingUser: null,
  exiting: false,          // Exit was pressed: don't reconnect
  pres: { phase: "ready", t0: 0, timer: 0, captions: [], result: null, sessionId: null },
  iv: { phase: "setup", scenario: "job_interview", details: "", messages: [], result: null, sessionId: null },
};

// ---------------------------------------------------------------- network

const net = {
  ws: null,
  retry: 0,
  connect() {
    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retry = 0;
      state.connected = true;
      onConnected();
    };
    ws.onclose = () => {
      if (this.ws !== ws) return;
      state.connected = false;
      state.serverMode = null;
      if (state.recording) { state.recording = false; audio.stop(true); }
      render();
      if (!state.exiting) setTimeout(() => this.connect(), Math.min(5000, 400 * 2 ** this.retry++));
    };
    ws.onmessage = (e) => (typeof e.data === "string" ? onEvent(JSON.parse(e.data)) : onSpeech(e.data));
    this.ws = ws;
  },
  send(obj) {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj));
  },
  sendAudio(buf) {
    if (this.ws && this.ws.readyState === 1) this.ws.send(buf);
  },
};

// ---------------------------------------------------------------- audio

const player = {
  ctx: null,
  next: 0,
  sources: new Set(),
  current: 0,
  cancelled: new Set(),
  enqueue(id, pcm) {
    if (!this.ctx || this.cancelled.has(id) || !pcm.length) return;
    if (state.releasedAt && id > state.releasedReply) {
      window.aiBuddyLatencyMs = Math.round(performance.now() - state.releasedAt);
      state.releasedAt = 0;
    }
    const f = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
    const buf = this.ctx.createBuffer(1, f.length, 24000);
    buf.copyToChannel(f, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    const t = Math.max(this.ctx.currentTime + 0.05, this.next);
    src.start(t);
    this.next = t + buf.duration;
    this.current = id;
    this.sources.add(src);
    src.onended = () => {
      this.sources.delete(src);
      if (!this.sources.size) this.setPlaying(false);
    };
    this.setPlaying(true);
  },
  stopAll() {
    for (const s of this.sources) { try { s.stop(); } catch { /* already stopped */ } }
    this.sources.clear();
    this.next = 0;
    if (this.current) this.cancelled.add(this.current);
    this.setPlaying(false);
  },
  setPlaying(on) {
    if (state.playing === on) return;
    state.playing = on;
    if (!on) state.lastPlaybackEnd = performance.now();
    render();
  },
};

const audio = {
  ctx: null,
  node: null,
  ready: null,
  async init() {
    if (!this.ready) {
      this.ready = this._init().catch((err) => {
        this.ready = null;
        throw err;
      });
    }
    await this.ready;
    if (this.ctx.state !== "running") await this.ctx.resume();
  },
  async _init() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const ctx = new AudioContext({ latencyHint: "interactive" });
    await ctx.audioWorklet.addModule("/static/capture-worklet.js");
    const node = new AudioWorkletNode(ctx, "capture", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    ctx.createMediaStreamSource(stream).connect(node).connect(mute).connect(ctx.destination);
    node.port.onmessage = (e) => this.onWorklet(e.data);
    this.ctx = ctx;
    this.node = node;
    player.ctx = ctx;
  },
  onWorklet(msg) {
    if (msg.type === "audio") net.sendAudio(msg.pcm);
    else if (msg.type === "stopped") { if (!this.discard) net.send({ type: "rec_stop" }); this.discard = false; }
    else if (msg.type === "level") onLevel(msg.db);
  },
  start() { this.node.port.postMessage("start"); },
  stop(discard = false) {
    this.discard = discard;
    if (this.node) this.node.port.postMessage("stop");
  },
};

function micError(err) {
  const blocked = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
  toast(blocked
    ? "Microphone access is blocked. Allow it for this page (lock icon in the address bar) and try again."
    : `Couldn't start the microphone: ${err && err.message ? err.message : err}`, "error", 8000);
}

// ---------------------------------------------------------------- talking

const pttMode = () => state.connected && (state.mode === "chat" || (state.mode === "interview" && state.iv.phase === "running"));
const toggleMode = () => state.mode === "presentation";
const serverBusy = () => ["transcribing", "thinking", "tool", "loading", "analyzing"].includes(state.server);

async function beginTalk() {
  if (state.recording) return;
  state.talkIntent = true;
  try {
    await audio.init();
  } catch (err) {
    state.talkIntent = false;
    return micError(err);
  }
  if (!state.talkIntent) return;           // released before the mic was ready
  player.stopAll();                         // barge-in: the buddy stops talking right away
  if (state.pendingUser) { state.pendingUser.remove(); state.pendingUser = null; }
  state.recording = true;
  net.send(state.mode === "presentation" ? { type: "rec_start", details: $("pres-topic").value.trim() } : { type: "rec_start" });
  audio.start();
  if (state.mode === "presentation") startPresentationClock();
  render();
}

function endTalk() {
  state.talkIntent = false;
  if (!state.recording) return;
  state.recording = false;
  audio.stop();
  state.releasedAt = performance.now();
  state.releasedReply = player.current;
  if (state.mode === "presentation") stopPresentationClock();
  else addPendingUser();
  render();
}

function cancelSpeech() {
  player.stopAll();
  net.send({ type: "cancel" });
}

// Experimental hands-free mode: energy-based voice detection on the 20 ms level reports.
const vad = {
  floor: -65,
  above: 0,
  below: 0,
  startedAt: 0,
  onLevel(db) {
    if (!state.recording) this.floor = db < this.floor ? this.floor * 0.6 + db * 0.4 : this.floor * 0.997 + db * 0.003;
    if (!state.handsFree || !pttMode()) return;
    const startAt = Math.max(this.floor + 15, -50);
    const stopAt = Math.max(this.floor + 8, -56);
    if (!state.recording) {
      const quiet = !state.playing && !serverBusy() && performance.now() - state.lastPlaybackEnd > 500;
      this.above = quiet && db > startAt ? this.above + 1 : 0;
      if (this.above >= 6) {        // ~120 ms of voice
        this.above = 0;
        this.below = 0;
        this.startedAt = performance.now();
        beginTalk();
      }
    } else {
      this.below = db < stopAt ? this.below + 1 : 0;
      const long = performance.now() - this.startedAt > 60000;
      if (this.below >= 45 || long) {  // ~900 ms of silence ends the turn
        this.below = 0;
        endTalk();
      }
    }
  },
};

function onLevel(db) {
  if (state.recording) $("meter-fill").style.height = `${Math.max(0, Math.min(100, ((db + 60) / 50) * 100))}%`;
  vad.onLevel(db);
}

// ---------------------------------------------------------------- server events

function onConnected() {
  render();
  if (state.mode === "chat" || state.mode === "presentation") startServerMode(state.mode);
  if (state.mode === "interview" && state.iv.phase !== "setup") {
    state.iv.phase = "setup";
    toast("Lost the connection, so the practice conversation ended. You can start again.");
    renderInterview();
  }
}

function startServerMode(mode, extra = {}) {
  if (mode === "chat") resetThread($("chat-thread"), true);
  net.send({ type: "start", mode, ...extra });
  state.serverMode = mode;
}

function onEvent(ev) {
  switch (ev.type) {
    case "models":
      state.models = ev.models;
      state.warm = ev.warm;
      renderModels();
      if (ev.error && ev.error !== state.lastModelError) toast(ev.error, "error", 8000);
      state.lastModelError = ev.error;
      break;
    case "session":
      if (ev.mode === "presentation" && ev.id) state.pres.sessionId = ev.id;
      if (ev.mode === "interview") state.iv.sessionId = ev.id;
      break;
    case "state":
      state.server = ev.state;
      state.detail = ev.detail || "";
      break;
    case "user_partial":
      showPartialUser(ev.text);
      break;
    case "user_text":
      fillPendingUser(ev.text);
      break;
    case "reply_start":
      if (!ev.readout) startReply(ev.id, ev.greeting);
      break;
    case "tool_start":
      onToolStart(ev);
      break;
    case "tool_end":
      onToolEnd(ev);
      break;
    case "confirm_request":
      onConfirmRequest(ev);
      break;
    case "confirm_end":
      onConfirmEnd(ev);
      break;
    case "reply_delta":
      appendReply(ev.id, ev.text);
      break;
    case "reply_end":
      endReply(ev);
      break;
    case "caption":
      addCaption(ev);
      break;
    case "metrics":
      onMetrics(ev);
      break;
    case "feedback_delta":
      onFeedbackDelta(ev);
      break;
    case "feedback_done":
      onFeedbackDone(ev);
      break;
    case "feedback_cancelled":
      onFeedbackCancelled(ev);
      break;
    case "notice":
      onNotice(ev.text);
      break;
    case "error":
      dropPendingUser();
      toast(ev.text, "error", 9000);
      break;
  }
  render();
}

function onSpeech(buf) {
  const id = new DataView(buf).getUint32(0, true);
  player.enqueue(id, new Int16Array(buf, 4, (buf.byteLength - 4) >> 1));
}

function onNotice(text) {
  dropPendingUser();
  toast(text);
}

// The server gave up on a practice session's feedback: nothing to grade, or the coach failed.
function onFeedbackCancelled(ev) {
  const p = state.pres;
  const iv = state.iv;
  if (ev.session_id && ev.session_id === p.sessionId && p.result && !p.result.done) {
    if (p.result.metrics) Object.assign(p.result, { done: true, failed: ev.text });
    else { p.phase = "ready"; p.result = null; }
    renderPresentation();
  }
  if (ev.session_id && ev.session_id === iv.sessionId && iv.result && !iv.result.done) {
    if (iv.result.metrics) Object.assign(iv.result, { done: true, failed: ev.text });
    else { iv.phase = "setup"; iv.result = null; }
    renderInterview();
  }
  toast(ev.text, ev.error ? "error" : "info", ev.error ? 9000 : 4500);
}

// ---------------------------------------------------------------- scrolling

// The page scrolls and the dock floats over its bottom (main has room for it). New content keeps the view at the
// bottom only while you're already there: scroll up and it stays put, with a button to jump back down.
const scroller = {
  follow: true,
  atBottom() {
    const doc = document.documentElement;
    return doc.scrollHeight - (window.scrollY + window.innerHeight) < 48;
  },
  onScroll() {
    this.follow = this.atBottom();
    if (this.follow) $("jump-latest").hidden = true;
  },
  // After adding or growing content. `force`: the user just did something (talked, opened the chat).
  update(force = false) {
    if (force) this.follow = true;
    if (this.follow) {
      window.scrollTo(0, document.documentElement.scrollHeight);
      $("jump-latest").hidden = true;
    } else if (state.mode === "chat" || state.mode === "interview") {
      $("jump-latest").hidden = false;
    }
  },
  bind() {
    window.addEventListener("scroll", () => this.onScroll(), { passive: true });
    // Scrolling up by hand stops following at once, before the next streamed word could pull the view back.
    window.addEventListener("wheel", (e) => { if (e.deltaY < 0) this.follow = false; }, { passive: true });
    window.addEventListener("keydown", (e) => {
      if (["PageUp", "ArrowUp", "Home"].includes(e.key) && !isTyping(e.target)) this.follow = false;
    });
    $("jump-latest").addEventListener("click", () => this.update(true));
  },
};

// ---------------------------------------------------------------- conversation threads

const activeThread = () => (state.serverMode === "interview" ? $("iv-thread") : $("chat-thread"));
const counterpart = () => (state.serverMode === "interview" ? state.config.scenarios[state.iv.scenario].counterpart : state.config.persona);

function resetThread(thread, withEmpty) {
  thread.replaceChildren();
  if (withEmpty) thread.append(chatEmpty);
  state.replies.clear();
}

function bubble(role, text, who) {
  const msg = el("div", { class: `msg ${role === "user" ? "user" : "buddy"}` },
    who ? el("div", { class: "who" }, who) : null,
    el("div", { class: "bubble" }, text));
  const thread = activeThread();
  thread.querySelector(".empty")?.remove();
  thread.append(msg);
  scroller.update(role === "user");     // your own words always bring the view down
  return msg;
}

function addPendingUser() {
  dropPendingUser();
  state.pendingUser = bubble("user", "…");
  state.pendingUser.classList.add("pending");
}

function dropPendingUser() {
  if (state.pendingUser) state.pendingUser.remove();
  state.pendingUser = null;
}

function showPartialUser(text) {
  if (!state.pendingUser) return;
  state.pendingUser.querySelector(".bubble").textContent = `${text} …`;
  scroller.update();
}

function fillPendingUser(text) {
  const msg = state.pendingUser || bubble("user", "");
  msg.classList.remove("pending");
  msg.querySelector(".bubble").textContent = text;
  state.pendingUser = null;
  if (state.serverMode === "interview") state.iv.messages.push({ role: "user", text });
}

function startReply(id, greeting = false) {
  const msg = bubble("buddy", "", counterpart());
  msg.querySelector(".bubble").classList.add("typing");
  if (greeting) msg.classList.add("greeting");
  state.replies.set(id, { msg, text: "", tools: new Map() });
}

// A tool Kira is using, shown as a line inside her message: "Searching the web · Calgary weather tomorrow".
function onToolStart(ev) {
  const r = state.replies.get(ev.id);
  if (!r) return;
  let box = r.msg.querySelector(".tools");
  if (!box) {
    box = el("div", { class: "tools" });
    r.msg.querySelector(".bubble").before(box);
  }
  const row = el("div", { class: "tool running" }, el("span", { class: "spinner" }),
    el("span", { class: "tool-label" }, ev.label), ev.detail ? el("span", { class: "tool-detail" }, ev.detail) : null);
  box.append(row);
  r.tools.set(ev.call_id, row);
  scroller.update();
}

function onToolEnd(ev) {
  const r = state.replies.get(ev.id);
  const row = r && r.tools.get(ev.call_id);
  if (!row) return;
  row.classList.remove("running");
  row.classList.toggle("failed", !ev.ok);
  row.querySelector(".spinner")?.remove();
  row.prepend(el("span", { class: "tool-mark", "aria-hidden": "true" }, ev.ok ? "✓" : "!"));
  if (ev.summary) row.append(el("span", { class: "tool-summary" }, ev.summary));
}

// A skill that changes something asks first: a card with the question, answered by voice or with the buttons.
function onConfirmRequest(ev) {
  const r = state.replies.get(ev.id);
  if (!r) return;
  const answer = (allow) => net.send({ type: "confirm", call_id: ev.call_id, allow });
  const card = el("div", { class: "confirm", "data-call": ev.call_id },
    el("div", { class: "confirm-q" }, ev.text),
    el("div", { class: "confirm-detail muted small" }, `${ev.skill} · ${ev.tool}`),
    el("div", { class: "row" },
      el("button", { type: "button", class: "btn primary", onclick: () => answer(true) }, "Yes, go ahead"),
      el("button", { type: "button", class: "btn", onclick: () => answer(false) }, "No")),
    el("div", { class: "muted small" }, "Or hold Space and say yes or no."));
  r.msg.append(card);
  scroller.update();
}

function onConfirmEnd(ev) {
  const card = document.querySelector(`.confirm[data-call="${CSS.escape(ev.call_id)}"]`);
  if (!card) return;
  card.querySelector(".row")?.remove();
  card.querySelector(".muted.small:last-child")?.remove();
  card.classList.add(ev.allowed ? "allowed" : "declined");
  card.append(el("div", { class: "confirm-result" }, ev.allowed ? "✓ You said yes" : "✕ You said no"));
}

function sourceList(sources) {
  if (!sources || !sources.length) return null;
  const items = sources.slice(0, 8).map((src) => {
    if (src.url) {
      let host = "";
      try { host = new URL(src.url).hostname.replace(/^www\./, ""); } catch { /* not a URL */ }
      return el("li", {}, el("a", { href: src.url, target: "_blank", rel: "noopener noreferrer" }, src.title || host),
        host ? el("span", { class: "muted" }, ` · ${host}`) : null);
    }
    return el("li", {}, el("button", { type: "button", class: "link", onclick: () => openNoteFile(src.path) }, src.title || src.path));
  });
  return el("details", { class: "sources" }, el("summary", {}, `Sources (${sources.length})`), el("ul", {}, items));
}

async function openNoteFile(path) {
  try {
    await api("/api/knowledge/open", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }) });
  } catch (err) {
    toast(`Couldn't open the file: ${err.message}`, "error");
  }
}

function appendReply(id, delta) {
  const r = state.replies.get(id);
  if (!r) return;
  r.text += delta;
  r.msg.querySelector(".bubble").textContent = stripEmoji(r.text).trimStart();
  scroller.update();
}

function endReply(ev) {
  const r = state.replies.get(ev.id);
  state.replies.delete(ev.id);
  if (!r) return;
  const b = r.msg.querySelector(".bubble");
  b.classList.remove("typing");
  if (!ev.text && !r.text.trim() && !r.tools.size) { r.msg.remove(); return; }
  if (ev.interrupted) {
    b.textContent = ev.text || stripEmoji(r.text);
    r.msg.append(el("div", { class: "meta" }, "interrupted"));
  }
  const sources = sourceList(ev.sources);
  if (sources) r.msg.append(sources);
  scroller.update();
  if (state.serverMode === "interview") state.iv.messages.push({ role: "assistant", text: b.textContent });
}

// ---------------------------------------------------------------- presentation

function startPresentationClock() {
  const p = state.pres;
  p.phase = "recording";
  p.captions = [];
  p.result = null;
  p.t0 = performance.now();
  $("pres-captions").replaceChildren();
  $("pres-live-hint").hidden = false;
  clearInterval(p.timer);
  p.timer = setInterval(() => { $("pres-timer").textContent = fmtClock((performance.now() - p.t0) / 1000); }, 250);
  $("pres-timer").textContent = "0:00";
  renderPresentation();
}

function stopPresentationClock() {
  clearInterval(state.pres.timer);
  state.pres.phase = "processing";
  state.pres.result = { metrics: null, md: "", done: false };
  renderPresentation();
}

function addCaption(ev) {
  state.pres.captions.push(ev.text);
  $("pres-live-hint").hidden = true;
  $("pres-captions").append(el("p", {}, el("span", { class: "t" }, fmtClock(ev.start_s)), ev.text));
  scroller.update();
}

function renderPresentation() {
  const p = state.pres;
  $("pres-intro").hidden = p.phase !== "ready";
  $("pres-live").hidden = !(p.phase === "recording" || (p.phase === "processing" && !p.result?.metrics));
  const results = $("pres-results");
  if (p.result && (p.phase === "processing" || p.phase === "results")) {
    renderResults(results, p.result, {
      transcript: p.captions.join(" "),
      actions: [
        ["Read the top tips aloud", () => readTips(p.result.md), () => p.result.done],
        ["New talk", () => { p.phase = "ready"; p.result = null; renderPresentation(); render(); }, () => p.result.done],
      ],
    });
  } else {
    results.replaceChildren();
  }
}

// ---------------------------------------------------------------- interview

let chatEmpty;

function buildScenarios() {
  const box = $("iv-scenarios");
  for (const [key, s] of Object.entries(state.config.scenarios)) {
    box.append(el("button", { type: "button", class: "scenario", role: "radio", "data-key": key,
      "aria-checked": key === state.iv.scenario ? "true" : "false",
      onclick: () => selectScenario(key) }, el("strong", {}, s.title), el("span", {}, s.blurb)));
  }
  selectScenario(state.iv.scenario);
}

function selectScenario(key) {
  state.iv.scenario = key;
  for (const b of $("iv-scenarios").children) b.setAttribute("aria-checked", b.dataset.key === key ? "true" : "false");
  const s = state.config.scenarios[key];
  $("iv-details").placeholder = s.placeholder;
  $("iv-details-label").textContent = key === "custom" ? "Describe the scenario" : "Details (optional)";
}

async function startInterview(e) {
  e.preventDefault();
  if (state.iv.scenario === "custom" && !$("iv-details").value.trim()) {
    $("iv-details").focus();
    return toast("Describe the scenario first.");
  }
  try { await audio.init(); } catch (err) { return micError(err); }
  const iv = state.iv;
  iv.details = $("iv-details").value.trim();
  iv.messages = [];
  iv.result = null;
  iv.phase = "running";
  resetThread($("iv-thread"), false);
  const s = state.config.scenarios[iv.scenario];
  $("iv-title").textContent = s.title;
  $("iv-sub").textContent = iv.details ? `· ${iv.details}` : "";
  startServerMode("interview", { scenario: iv.scenario, details: iv.details });
  renderInterview();
  render();
}

function endInterview() {
  if (state.recording) endTalk();
  cancelSpeechLocal();
  state.iv.phase = "ending";
  state.iv.result = { metrics: null, md: "", done: false };
  net.send({ type: "end" });
  renderInterview();
  render();
}

function cancelSpeechLocal() { player.stopAll(); }

function renderInterview() {
  const iv = state.iv;
  const firstResults = iv.result && !iv.resultsShown && (iv.phase === "ending" || iv.phase === "results");
  $("iv-setup").hidden = iv.phase !== "setup";
  $("iv-running").hidden = iv.phase === "setup";
  $("iv-end").hidden = iv.phase !== "running";
  const results = $("iv-results");
  if (iv.result && (iv.phase === "ending" || iv.phase === "results")) {
    renderResults(results, iv.result, {
      actions: [
        ["Read the top tips aloud", () => readTips(iv.result.md), () => iv.result.done],
        ["Practice again", () => { iv.phase = "setup"; iv.result = null; renderInterview(); render(); }, () => iv.result.done],
      ],
    });
    if (firstResults) results.scrollIntoView({ behavior: "smooth", block: "start" });
    iv.resultsShown = true;
  } else {
    iv.resultsShown = false;
    results.replaceChildren();
  }
}

// ---------------------------------------------------------------- feedback results

function onMetrics(ev) {
  const kind = ev.session_id === state.pres.sessionId ? "presentation" : ev.session_id === state.iv.sessionId ? "interview" : null;
  const r = kind === "presentation" ? state.pres.result : kind === "interview" ? state.iv.result : null;
  if (!r) return;
  r.metrics = ev.metrics;
  r.sessionId = ev.session_id;
  if (kind === "presentation") { state.pres.phase = "results"; renderPresentation(); }
  else { state.iv.phase = "results"; renderInterview(); }
}

let fbFrame = 0;
function onFeedbackDelta(ev) {
  const r = [state.pres.result, state.iv.result].find((x) => x && x.sessionId === ev.session_id);
  if (!r) return;
  r.md += ev.text;
  if (!fbFrame) fbFrame = requestAnimationFrame(() => {
    fbFrame = 0;
    for (const box of document.querySelectorAll(`.feedback[data-session="${CSS.escape(ev.session_id)}"]`)) {
      box.innerHTML = renderMarkdown(r.md);
      box.lastElementChild?.classList.add("typing");
    }
  });
}

function onFeedbackDone(ev) {
  const r = [state.pres.result, state.iv.result].find((x) => x && x.sessionId === ev.session_id);
  if (r) {
    r.md = ev.markdown;
    r.done = true;
    if (r === state.pres.result) renderPresentation(); else renderInterview();
  }
  const visible = (r === state.pres.result && state.mode === "presentation") || (r === state.iv.result && state.mode === "interview");
  if (!visible) toast("Your feedback is ready. Find it in History.");
}

function tile(label, value, unit, sub) {
  return el("div", { class: "tile" }, el("div", { class: "label" }, label),
    el("div", { class: "value" }, String(value), unit ? el("small", {}, unit) : null),
    sub ? el("div", { class: "sub" }, sub) : null);
}

function metricTiles(m) {
  const pace = { good: "comfortable pace", fast: "a bit fast", slow: "a bit slow" }[m.pace] || "";
  const f = m.fillers;
  const p = m.pauses;
  const lengthSub = m.answers ? `${m.answers.length} answers, avg ${m.avg_answer_s} s` : `${m.words} words`;
  return el("div", { class: "tiles" },
    tile("Pace", m.wpm, "wpm", `${pace} (${m.target_wpm[0]}–${m.target_wpm[1]})`),
    tile("Filler words", f.total, null, `${f.per_min} per minute`),
    tile("Long pauses", p.count, null, p.count ? `longest ${p.longest_s} s` : `none over ${p.threshold_s} s`),
    tile("Speaking time", fmtClock(m.speaking_time_s), null, lengthSub));
}

function fillerChips(m) {
  const entries = Object.entries(m.fillers.by_word);
  return el("div", { class: "chips", "aria-label": "Filler words" },
    entries.length ? entries.map(([w, n]) => el("span", { class: "chip" }, `“${w}” `, el("b", {}, `×${n}`)))
      : el("span", { class: "chip" }, "No filler words detected"));
}

function renderResults(container, r, opts = {}) {
  const fb = el("div", { class: "feedback", "data-session": r.sessionId || "" });
  if (r.md) {
    fb.innerHTML = renderMarkdown(r.md);
    if (!r.done) fb.lastElementChild?.classList.add("typing");
  } else if (r.done) {
    fb.append(el("p", { class: "muted" }, r.failed || "No feedback was saved for this session."));
  } else {
    const label = !r.metrics ? (state.detail || "Finishing the transcript…")
      : state.detail || "Writing your feedback…";
    fb.append(el("div", { class: "placeholder" }, el("span", { class: "spinner" }), el("span", { class: "fb-status" }, label)));
  }
  const parts = [];
  if (r.metrics) parts.push(metricTiles(r.metrics), fillerChips(r.metrics));
  parts.push(fb);
  if (opts.actions) {
    parts.push(el("div", { class: "row" }, opts.actions.map(([label, fn, enabled], i) =>
      el("button", { class: `btn${i === 0 ? "" : " primary"}`, type: "button", disabled: enabled && !enabled(), onclick: fn }, label))));
  }
  if (opts.transcript) {
    parts.push(el("details", { class: "transcript" }, el("summary", {}, "Transcript"), el("p", {}, opts.transcript)));
  }
  container.replaceChildren(el("div", { class: "results" }, parts));
}

async function readTips(md) {
  const text = sectionForSpeech(md, "improve") || sectionForSpeech(md, "overall");
  if (!text) return;
  try { await audio.init(); } catch (err) { return micError(err); }
  player.stopAll();
  net.send({ type: "speak", text });
}

// ---------------------------------------------------------------- history

async function renderHistory() {
  $("history-path").textContent = `Transcripts and feedback are saved in ${state.config.data_dir}`;
  let sessions;
  try {
    sessions = await api("/api/sessions");
  } catch (err) {
    return toast(`Couldn't load history: ${err.message}`, "error");
  }
  const practice = sessions.filter((s) => s.wpm != null).reverse();   // oldest first
  const trends = $("history-trends");
  if (practice.length >= 2) {
    trends.replaceChildren();
    requestAnimationFrame(() => renderTrends(trends, practice, state.config.practice));
  } else if (practice.length === 1) {
    const s = practice[0];
    trends.replaceChildren(el("p", { class: "muted" }, "Your first practice session. Trends appear after the second one."),
      el("div", { class: "tiles" },
        tile("Pace", s.wpm, "wpm", fmtWhen(s.started_at)),
        tile("Filler words", s.fillers_per_min, "/min", s.title),
        tile("Long pauses", s.long_pauses, null, `≥ ${state.config.practice.long_pause_s} s`),
        tile("Words", s.words, null, "")));
  } else {
    trends.replaceChildren(el("div", { class: "card" }, el("h2", {}, "No practice sessions yet"),
      el("p", { class: "muted" }, "Do a presentation or a conversation practice and your pace, filler words, and pauses will be charted here.")));
  }

  const table = $("history-table");
  if (!sessions.length) return table.replaceChildren();
  const num = (v, suffix = "") => (v == null ? "–" : `${v}${suffix}`);
  table.replaceChildren(el("table", { class: "sessions" },
    el("thead", {}, el("tr", {},
      el("th", {}, "When"), el("th", {}, "Type"), el("th", { class: "num opt" }, "Length"),
      el("th", { class: "num" }, "Pace (wpm)"), el("th", { class: "num" }, "Fillers / min"),
      el("th", { class: "num" }, "Long pauses"), el("th", { class: "opt" }, "Feedback"))),
    el("tbody", {}, sessions.map((s) => el("tr", { tabindex: "0", onclick: () => openSession(s.id),
      onkeydown: (e) => { if (e.key === "Enter") openSession(s.id); } },
      el("td", {}, fmtWhen(s.started_at)),
      el("td", {}, s.title, s.mode === "chat" ? el("span", { class: "muted" }, ` · ${s.user_turns} turns`) : null),
      el("td", { class: "num opt" }, s.duration_s ? fmtClock(s.duration_s) : "–"),
      el("td", { class: "num" }, num(s.wpm)), el("td", { class: "num" }, num(s.fillers_per_min)),
      el("td", { class: "num" }, num(s.long_pauses)),
      el("td", { class: "opt" }, s.has_feedback ? "Yes" : "–"))))));
}

async function openSession(id) {
  const s = await api(`/api/sessions/${encodeURIComponent(id)}`);
  const body = $("detail-body");
  const who = s.scenario ? state.config.scenarios[s.scenario]?.counterpart : state.config.persona;
  const results = el("div");
  if (s.metrics || s.feedback_md) renderResults(results, { metrics: s.metrics, md: s.feedback_md || "", done: true, sessionId: s.id });
  const thread = el("div", { class: "thread" }, s.turns.map((t) =>
    el("div", { class: `msg ${t.role === "user" ? "user" : "buddy"}` },
      t.role === "user" ? null : el("div", { class: "who" }, who),
      el("div", { class: "bubble" }, t.text),
      t.interrupted ? el("div", { class: "meta" }, "interrupted") : null,
      t.meta ? sourceList(t.meta.sources) : null)));
  body.replaceChildren(
    el("div", { class: "detail-head" },
      el("div", {}, el("h1", {}, s.title), el("p", { class: "muted" }, fmtWhen(s.started_at), s.details ? ` · ${s.details}` : "",
        s.feedback_model ? ` · feedback by ${s.feedback_model}` : "")),
      el("div", { class: "row" },
        el("button", { class: "btn danger", type: "button", onclick: () => deleteSession(s.id) }, "Delete"),
        el("button", { class: "btn", type: "button", onclick: () => $("detail").close() }, "Close"))),
    results,
    s.mode === "presentation" ? el("details", { class: "transcript", open: !s.feedback_md }, el("summary", {}, "Transcript"),
      el("p", {}, s.turns.map((t) => t.text).join(" ")))
      : el("details", { class: "transcript", open: true }, el("summary", {}, "Conversation"), thread));
  $("detail").showModal();
}

async function deleteSession(id) {
  if (!confirm("Delete this session and its transcript?")) return;
  await api(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
  $("detail").close();
  renderHistory();
}

// ---------------------------------------------------------------- rendering

function renderModels() {
  if (!state.models) return;
  const labels = { stt: "Whisper", llm: state.config.persona, tts: "Voice", embed: "Notes search" };
  const box = $("models");
  box.replaceChildren(...Object.entries(state.models).map(([role, m]) => {
    const name = labels[role] || role;
    const device = m.device || "";
    const note = { ready: "", loading: "loading", standby: "standby", error: "error", unknown: "…" }[m.state] ?? m.state;
    return el("span", { class: "pill", "data-state": m.state, title: `${m.name} on the ${device}: ${m.state}` },
      el("span", { class: "dot" }), name, note ? el("span", { class: "state" }, note) : null);
  }));
}

function statusText() {
  if (!state.connected) return "Connecting to AI Buddy…";
  if (state.recording) return state.mode === "presentation" ? "Recording your talk. Press Stop when you're done." : "Listening…";
  if (state.server === "transcribing") return "Transcribing…";
  if (state.server === "thinking") return "Thinking…";
  if (state.server === "tool") return state.detail || "Looking it up…";
  if (state.server === "confirm") return "Waiting for your answer: say yes or no, or use the buttons.";
  if (state.server === "loading" || state.server === "analyzing") return state.detail || "Working…";
  if (state.playing) return `${counterpart()} is talking. Press Space to interrupt.`;
  if (!state.warm) return "Warming up the models…";
  if (state.mode === "presentation") return state.pres.phase === "results" ? "Ready for another take." : "Ready when you are.";
  return state.handsFree ? "Hands-free: just start talking." : "Hold Space or the button to talk.";
}

function render() {
  const dock = $("dock");
  const m = state.mode;
  const showDock = m === "chat" || m === "presentation" || (m === "interview" && state.iv.phase === "running");
  dock.hidden = !showDock;
  dock.dataset.state = state.recording ? "listening" : state.playing ? "speaking" : "idle";
  dock.dataset.busy = String(!state.recording && (serverBusy() || (!state.warm && state.connected)));
  dock.dataset.toggle = String(toggleMode());
  const talk = $("talk");
  const presBusy = m === "presentation" && (state.pres.phase === "processing" || (state.pres.result && !state.pres.result.done));
  talk.disabled = !state.connected || presBusy || (m === "presentation" && state.server === "analyzing" && !state.recording);
  if (toggleMode()) {
    talk.setAttribute("aria-label", state.recording ? "Stop" : "Start talk");
    setToggleLabel(talk, state.recording ? "Stop" : "Start talk");
  } else {
    talk.setAttribute("aria-label", "Hold to talk");
    setToggleLabel(talk, "");
  }
  $("status-text").textContent = statusText();
  const hint = m === "presentation" ? "" : state.playing ? "Esc stops the voice" : "";
  $("dock-hint").textContent = hint;
  $("handsfree-wrap").hidden = !(m === "chat" || (m === "interview" && state.iv.phase === "running"));
  const fbStatus = document.querySelector(".fb-status");
  if (fbStatus && state.detail) fbStatus.textContent = state.detail;
}

function setToggleLabel(btn, text) {
  let label = btn.querySelector(".label");
  if (!text) { label?.remove(); return; }
  if (!label) { label = el("span", { class: "label" }); btn.insertBefore(label, btn.querySelector(".ring")); }
  label.textContent = text;
}

// Practice tabs come from the speaking-coach skill; the empty chat says what the enabled skills can do.
const SKILL_PHRASES = { "web-search": "look things up online", notes: "check your notes", calculator: "do the math",
  clock: "tell the time", "speaking-coach": "coach you through a talk or an interview" };

function applySkills() {
  const modes = new Set(state.config.modes || ["chat"]);
  for (const b of document.querySelectorAll(".tabs button")) {
    if (b.dataset.mode !== "history") b.hidden = !modes.has(b.dataset.mode);
  }
  const on = state.config.skills.filter((s) => s.enabled && s.status === "ready");
  const phrases = on.map((s) => SKILL_PHRASES[s.name] || `use ${s.label}`);
  const list = phrases.length > 1 ? `${phrases.slice(0, -1).join(", ")} and ${phrases.at(-1)}` : phrases[0];
  $("chat-skills-line").textContent = list ? `I can ${list}.` : "Every skill is switched off; turn some on in Settings.";
  if (state.mode && state.mode !== "history" && !modes.has(state.mode)) switchMode("chat");
}

function switchMode(mode) {
  if (mode === state.mode) return;
  if (state.recording) {
    state.recording = false;
    state.talkIntent = false;
    audio.stop(true);
    if (state.mode === "presentation") { clearInterval(state.pres.timer); state.pres.phase = "ready"; }
  }
  cancelSpeechLocal();
  state.mode = mode;
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", String(b.dataset.mode === mode));
  for (const v of ["chat", "presentation", "interview", "history"]) $(`view-${v}`).hidden = v !== mode;
  try { localStorage.setItem("ai-buddy-mode", mode); } catch { /* storage unavailable */ }
  if (mode === "chat" && state.serverMode !== "chat") startServerMode("chat");
  if (mode === "presentation" && state.serverMode !== "presentation") {
    if (state.pres.phase !== "processing") { state.pres.phase = "ready"; state.pres.result = null; }
    startServerMode("presentation");
    renderPresentation();
  }
  if (mode === "interview" && state.serverMode !== "interview") { state.iv.phase = "setup"; renderInterview(); }
  if (mode === "history") renderHistory();
  if (mode === "chat" || mode === "interview") scroller.update(true);
  else window.scrollTo(0, 0);
  render();
}

// ---------------------------------------------------------------- settings

const LLM_NOTES = {
  "qwen3.5-9b-FLM": "The default. Reads Whisper's transcript of what you say; good with tools.",
  "gemma4-it-12b-FLM": "Reads Whisper's transcript too; reliable with tools, a little slower to answer.",
  "qwen3.6-moe-35b-a3b-FLM": "Accurate, but needs much more memory (22 GB) and takes longer to start answering.",
};
const STT_NOTES = {
  "whisper-v3-turbo-FLM": "Whisper Large v3 Turbo on the NPU (FastFlowLM): the default, light on power.",
  "Whisper-Large-v3-Turbo": "The same Whisper model on whisper.cpp, running on the CPU instead of the NPU. " +
    "In earlier tests it made fewer mistakes on real speech and kept more filler words.",
};

// The assistant is named after its voice (Bella by default).
function setPersona(name) {
  if (!name || !state.config) return;
  state.config.persona = name;
  document.querySelectorAll(".persona-name").forEach((e) => { e.textContent = name; });
  renderModels();
}

const settingsUI = {
  data: null,
  poll: 0,
  async open() {
    $("settings").showModal();
    await this.load();
  },
  async load() {
    try {
      this.data = await api("/api/settings");
    } catch (err) {
      return toast(`Couldn't load settings: ${err.message}`, "error");
    }
    const d = this.data;
    setPersona(d.persona);
    const option = (value, label, selected, isDefault) =>
      el("option", { value, selected }, isDefault ? `${label} — default` : label);
    $("set-voice").replaceChildren(...Object.entries(d.voices).map(([id, label]) =>
      option(id, label, id === d.voice, id === d.default_voice)));
    $("set-llm").replaceChildren(...d.llms.map((m) => option(m, m, m === d.llm, m === d.default_llm)));
    $("set-llm-note").textContent = LLM_NOTES[d.llm] || "";
    $("set-stt").replaceChildren(...d.stts.map((m) => option(m, m, m === d.stt, m === d.default_stt)));
    $("set-stt-note").textContent = STT_NOTES[d.stt] || "";
    $("set-models").textContent = `Language model: ${d.llm} (NPU) · Speech-to-text: ${d.stt} (${d.stt_device}) · ` +
      `Voice: ${d.tts} (CPU)` + (d.embed ? ` · Notes search: ${d.embed} (CPU)` : "") +
      ". Nothing leaves this laptop except web searches and the MCP servers you connect.";
    this.renderSkills(d.skills);
    $("pick-folder").hidden = !d.can_pick_folder;
    if (document.activeElement !== $("set-folder")) $("set-folder").value = d.knowledge.folder || "";
    this.renderIndex(d.knowledge);
  },
  renderSkills(skills) {
    const status = (s) => s.status === "error" ? `Couldn't connect: ${s.error}` :
      s.status === "starting" ? "Connecting…" : s.source === "mcp" && s.enabled && s.status === "off" ? "Not connected" : "";
    $("set-skills").replaceChildren(...skills.map((s) => {
      const asks = s.tools.filter((t) => t.confirm).length;
      const input = el("input", { type: "checkbox", checked: s.enabled, "aria-label": `${s.label} on or off` });
      input.addEventListener("change", () => this.toggleSkill(s.name, input.checked, input));
      return el("li", { class: "skill", "data-status": s.status },
        el("label", { class: "switch" }, input, el("span", { class: "slider", "aria-hidden": "true" })),
        el("div", { class: "skill-body" },
          el("div", { class: "skill-name" }, s.label,
            s.source !== "built-in" ? el("span", { class: "tag" }, s.source === "mcp" ? "MCP" : "yours") : null,
            asks ? el("span", { class: "tag asks", title: "Asks you before it changes anything" }, "asks first") : null),
          el("div", { class: "muted small" }, s.description),
          status(s) ? el("div", { class: "small skill-status" }, status(s)) : null));
    }));
  },
  async toggleSkill(name, on, input) {
    try {
      state.config = await api("/api/skills", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, enabled: on }) });
    } catch (err) {
      input.checked = !on;
      return toast(`Couldn't switch that skill: ${err.message}`, "error");
    }
    applySkills();
    this.renderSkills(state.config.skills);
    if (state.config.skills.some((s) => s.status === "starting")) setTimeout(() => this.load(), 1500);
  },
  renderIndex(k) {
    let text;
    if (!k.folder) text = "No folder chosen.";
    else if (!k.exists) text = "That folder doesn't exist.";
    else if (k.indexing) text = `Indexing… ${k.files} files so far`;
    else text = `${k.files} file${k.files === 1 ? "" : "s"}, ${k.chunks} passages indexed` +
      (k.last_indexed ? ` · updated ${new Date(k.last_indexed).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}` : "");
    if (k.error) text += ` · ${k.error}`;
    if (k.folder && k.exists && k.chunks) {
      const e = k.embeddings || "off";
      text += e === "ready" ? " · search by meaning and keywords" :
        e === "indexing" || e === "pending" ? ` · understanding them (${k.embedded_chunks || 0}/${k.chunks})…` :
        e === "off" ? " · keyword search only" : " · keyword search only (the embedding model isn't available)";
    }
    $("index-status").textContent = text;
    $("index-errors").replaceChildren(...(k.errors || []).map((e) => el("li", {}, `${e.file}: ${e.error}`)));
    clearTimeout(this.poll);
    const busy = k.indexing || (k.chunks && ["indexing", "pending"].includes(k.embeddings));
    if (busy && $("settings").open) this.poll = setTimeout(() => this.refreshIndex(), 1000);
  },
  async refreshIndex() {
    try { this.renderIndex(await api("/api/knowledge")); } catch { /* try again on the next open */ }
  },
  async save(body) {
    try {
      this.data = await api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch (err) {
      toast(`Couldn't save: ${err.message}`, "error");
      return false;
    }
    await this.load();
    return true;
  },
  // One voice sample at a time: a new one (or Stop, or closing Settings) stops the one playing.
  sample: null,
  previewVoice() {
    this.stopSample();
    const a = new Audio(`/api/voice/preview?voice=${encodeURIComponent($("set-voice").value)}`);
    this.sample = a;
    $("stop-voice").disabled = false;
    const done = () => {
      if (this.sample !== a) return;
      this.sample = null;
      $("stop-voice").disabled = true;
    };
    a.addEventListener("ended", done);
    a.addEventListener("error", done);
    a.play().catch((err) => {
      done();
      if (err.name !== "AbortError") toast(`Couldn't play the sample: ${err.message}`, "error");
    });
  },
  stopSample() {
    const a = this.sample;
    this.sample = null;
    $("stop-voice").disabled = true;
    if (!a) return;
    a.pause();
    a.removeAttribute("src");   // also cancels a sample that's still being synthesized
    a.load();
  },
  async pickFolder() {
    let res;
    try {
      res = await api("/api/knowledge/pick", { method: "POST" });
    } catch (err) {
      return toast(`Couldn't open the folder chooser: ${err.message}`, "error");
    }
    if (res.path) {
      $("set-folder").value = res.path;
      if (await this.save({ knowledge_folder: res.path })) toast(`Knowledge folder set to ${res.path}`);
    }
  },
  bind() {
    $("open-settings").addEventListener("click", () => this.open());
    $("close-settings").addEventListener("click", () => $("settings").close());
    $("settings").addEventListener("click", (e) => { if (e.target === $("settings")) $("settings").close(); });
    $("settings").addEventListener("close", () => this.stopSample());
    $("set-voice").addEventListener("change", (e) => {
      const voice = e.target.value;
      this.stopSample();
      // Play the new voice once it's saved, unless Settings was closed or another voice picked meanwhile.
      this.save({ voice }).then((ok) => ok && $("settings").open && $("set-voice").value === voice && this.previewVoice());
    });
    $("preview-voice").addEventListener("click", () => this.previewVoice());
    $("stop-voice").addEventListener("click", () => this.stopSample());
    $("set-llm").addEventListener("change", async (e) => {
      if (await this.save({ llm: e.target.value })) toast(`Switching to ${e.target.value}. It takes a few seconds to load.`);
    });
    $("set-stt").addEventListener("change", async (e) => {
      if (await this.save({ stt: e.target.value })) toast(`Switching speech-to-text to ${e.target.value}.`);
    });
    $("pick-folder").addEventListener("click", () => this.pickFolder());
    $("use-folder").addEventListener("click", async () => {
      const path = $("set-folder").value.trim();
      if (await this.save({ knowledge_folder: path })) toast(path ? `Knowledge folder set to ${path}` : "Knowledge folder cleared");
    });
    $("set-folder").addEventListener("keydown", (e) => { if (e.key === "Enter") $("use-folder").click(); });
    $("reindex").addEventListener("click", async () => {
      try {
        this.renderIndex(await api("/api/knowledge/reindex", { method: "POST" }));
      } catch (err) {
        toast(`Couldn't re-index: ${err.message}`, "error");
      }
    });
  },
};

// ---------------------------------------------------------------- exit

// Exit: the server stops every session, unloads all models from Lemonade and shuts down; then this window closes
// (the app window is also closed by the launcher, which opened it).
async function exitApp() {
  if (!confirm(`Exit AI Buddy? ${state.config.persona} stops and all models are unloaded from Lemonade.`)) return;
  state.exiting = true;
  player.stopAll();
  settingsUI.stopSample();
  if (state.recording) { state.recording = false; state.talkIntent = false; audio.stop(true); }
  showStopped("Stopping AI Buddy…", "Unloading the models from Lemonade.");
  let res = null;
  try {
    res = await api("/api/exit", { method: "POST" });
  } catch (err) {
    showStopped("AI Buddy has stopped.", `Couldn't confirm that the models were unloaded (${err.message}).`);
  }
  if (res) {
    showStopped("AI Buddy has stopped.", res.unloaded ? "The models are unloaded. You can close this window."
      : "Lemonade didn't unload the models; check it with `lemonade status`. You can close this window.");
  }
  net.ws?.close();
  setTimeout(() => window.close(), 300);
}

function showStopped(title, text) {
  for (const d of document.querySelectorAll("dialog[open]")) d.close();
  $("stopped-title").textContent = title;
  $("stopped-text").textContent = text;
  $("stopped").hidden = false;
}

// ---------------------------------------------------------------- toasts

function toast(text, kind = "info", ms = 4500) {
  const t = el("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" }, el("span", {}, text),
    el("button", { type: "button", "aria-label": "Dismiss", onclick: () => t.remove() }, "✕"));
  $("toasts").append(t);
  setTimeout(() => t.remove(), ms);
}

// ---------------------------------------------------------------- wiring

const isTyping = (t) => t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);

function bindUI() {
  for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => switchMode(b.dataset.mode));

  const talk = $("talk");
  talk.addEventListener("pointerdown", (e) => {
    if (talk.disabled || e.button !== 0) return;
    if (toggleMode()) return;
    talk.setPointerCapture(e.pointerId);
    beginTalk();
  });
  const release = () => { if (!toggleMode()) endTalk(); };
  talk.addEventListener("pointerup", release);
  talk.addEventListener("pointercancel", release);
  talk.addEventListener("click", () => {
    if (!toggleMode() || talk.disabled) return;
    if (state.recording) endTalk(); else beginTalk();
  });
  for (const kind of ["keydown", "keyup"]) talk.addEventListener(kind, (e) => { if (e.code === "Space") e.preventDefault(); });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.playing) { cancelSpeech(); return; }
    if (e.code !== "Space" || isTyping(e.target) || $("detail").open || $("settings").open) return;
    if (pttMode()) {
      e.preventDefault();
      if (!e.repeat) beginTalk();
    }
  });
  document.addEventListener("keyup", (e) => {
    if (e.code !== "Space" || isTyping(e.target)) return;
    if (pttMode()) { e.preventDefault(); endTalk(); }
  });
  window.addEventListener("blur", () => { if (state.recording && !toggleMode() && !state.handsFree) endTalk(); });

  $("exit-app").addEventListener("click", exitApp);
  scroller.bind();
  $("iv-setup").addEventListener("submit", startInterview);
  $("iv-end").addEventListener("click", endInterview);
  $("handsfree").addEventListener("change", async (e) => {
    state.handsFree = e.target.checked;
    if (state.handsFree) {
      try { await audio.init(); } catch (err) { e.target.checked = state.handsFree = false; return micError(err); }
      toast("Hands-free is on: talk normally and pause to let me answer. It's experimental; use Space if it misfires.");
    }
    render();
  });
  $("detail").addEventListener("click", (e) => { if (e.target === $("detail")) $("detail").close(); });
}

async function main() {
  chatEmpty = $("chat-empty");
  state.config = await api("/api/config");
  setPersona(state.config.persona);
  buildScenarios();
  bindUI();
  settingsUI.bind();
  let saved = "chat";
  try { saved = localStorage.getItem("ai-buddy-mode") || "chat"; } catch { /* storage unavailable */ }
  state.mode = null;
  const usable = [...(state.config.modes || ["chat"]), "history"];
  switchMode(usable.includes(saved) ? saved : "chat");
  applySkills();
  net.connect();
}

main().catch((err) => toast(`Failed to start: ${err.message}`, "error", 20000));
