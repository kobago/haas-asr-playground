// Haas Playground SPA: real-time captions (WebSocket), record-then-transcribe and
// file upload (POST /api/transcribe). Mic audio is resampled to 16 kHz 16-bit PCM
// in an AudioWorklet (capture-worklet.js).

const $ = (id) => document.getElementById(id);
const RATE = 16000, FRAME_MS = 100, SPEECH_RMS = 0.012;

// ------------------------------------------------------------------ helpers

const fmtClock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

function status(text, error = false) {
  const el = $("status");
  el.hidden = !text;
  el.textContent = text || "";
  el.className = "status" + (error ? " error" : "");
}

function row(start, text) {
  const div = document.createElement("div");
  div.className = "row";
  const time = document.createElement("time");
  time.textContent = fmtClock(start);
  const p = document.createElement("p");
  p.textContent = text;
  div.append(time, p);
  return div;
}

function stats(el, items) {
  el.hidden = false;
  el.replaceChildren(...items.map(([value, label]) => {
    const d = document.createElement("div");
    const v = document.createElement("span");
    v.textContent = value;
    const l = document.createElement("small");
    l.textContent = label;
    d.append(v, l);
    return d;
  }));
}

function color(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Scrolling level meter: one bar per sample() call.
class Meter {
  constructor(canvas) { this.canvas = canvas; this.levels = []; this.draw(); }
  push(rms) { this.levels.push(rms); if (this.levels.length > 400) this.levels.shift(); this.draw(); }
  reset() { this.levels = []; this.draw(); }
  draw() {
    const c = this.canvas, dpr = devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext("2d");
    g.scale(dpr, dpr);
    g.fillStyle = color("--rule");
    g.fillRect(0, h / 2, w, 1);
    const step = 5, n = Math.floor(w / step), shown = this.levels.slice(-n);
    shown.forEach((rms, i) => {
      const amp = Math.max(1, Math.min(1, rms * 7) * (h / 2 - 2));
      g.fillStyle = rms > SPEECH_RMS ? color("--accent") : color("--wave");
      g.fillRect(w - (shown.length - i) * step, h / 2 - amp, 3, amp * 2);
    });
  }
}

// Open the microphone; onFrame gets each 100 ms Int16 PCM frame (ArrayBuffer).
async function openMic(onFrame, meter) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("/static/capture-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "capture", {
    processorOptions: { targetRate: RATE, frameMs: FRAME_MS },
  });
  node.port.onmessage = (e) => onFrame(e.data);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(node);
  source.connect(analyser);
  const mute = ctx.createGain();
  mute.gain.value = 0;
  node.connect(mute).connect(ctx.destination); // keep the graph running without echo
  const buf = new Float32Array(analyser.fftSize);
  const t0 = performance.now();
  const timer = setInterval(() => {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (const x of buf) sum += x * x;
    meter.push(Math.sqrt(sum / buf.length));
  }, 50);
  return {
    elapsed: () => (performance.now() - t0) / 1000,
    close() {
      clearInterval(timer);
      node.port.onmessage = null;
      node.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
    },
  };
}

function micError(err) {
  if (!window.isSecureContext) return "マイクは HTTPS でのみ使えます。https:// の URL で開いてください。";
  return `マイクを使用できません: ${err.message || err}`;
}

// ------------------------------------------------------------------ models

let models = [], model = null;

function savedModel() {
  try { return localStorage.getItem("model"); } catch { return null; }
}

function setModel(id) {
  model = id;
  try { localStorage.setItem("model", id); } catch { /* storage unavailable */ }
  const m = models.find((x) => x.id === id);
  $("info").textContent = m ? `${m.device} · ${m.precision}` : "";
}

async function postAudio(blob, id = model) {
  const res = await fetch(`/api/transcribe?model=${encodeURIComponent(id)}`, { method: "POST", body: blob });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

function showResult(out, statsEl, r) {
  stats(statsEl, [
    [fmtClock(r.duration), "音声の長さ"],
    [`${(r.elapsed * 1000).toFixed(0)} ms`, "認識時間"],
    [r.rtfx ? `×${r.rtfx.toFixed(0)}` : "–", "実時間比 (RTFx)"],
  ]);
  out.replaceChildren();
  if (!r.segments.length) {
    out.innerHTML = '<p class="empty">音声から文字を認識できませんでした。</p>';
    return;
  }
  const full = document.createElement("div");
  full.className = "full";
  const p = document.createElement("p");
  p.style.margin = "0";
  p.textContent = r.text;
  const copy = document.createElement("button");
  copy.textContent = "全文をコピー";
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(r.text); copy.textContent = "コピーしました"; }
    catch { copy.textContent = "コピーできませんでした"; }
  };
  full.append(p, copy);
  out.append(full, ...r.segments.map((s) => row(s.start, s.text)));
}

// Run the same audio through every model, one after another so CPU models do not
// compete for cores, and show the results side by side.
async function compareAll(out, statsEl, blob) {
  statsEl.hidden = true;
  const cards = document.createElement("div");
  cards.className = "cards";
  const slots = models.map((m) => {
    const card = document.createElement("div");
    card.className = "card";
    const h = document.createElement("h3");
    h.textContent = m.label;
    const meta = document.createElement("small");
    meta.textContent = `${m.device} · ${m.precision}`;
    h.append(meta);
    const p = document.createElement("p");
    p.className = "wait";
    p.textContent = "待機中…";
    card.append(h, p);
    cards.append(card);
    return { m, card, h, p };
  });
  out.replaceChildren(cards);
  for (const { m, card, h, p } of slots) {
    p.textContent = "認識中…";
    try {
      const r = await postAudio(blob, m.id);
      const t = document.createElement("small");
      t.textContent = `${(r.elapsed * 1000).toFixed(0)} ms · RTFx ×${r.rtfx.toFixed(0)}`;
      h.append(t);
      p.className = "";
      p.textContent = r.text || "（認識結果なし）";
    } catch (err) {
      card.classList.add("error");
      p.textContent = `失敗しました: ${err.message}`;
    }
  }
}

function wav(chunks) {
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const str = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, RATE, true); v.setUint32(28, RATE * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, n * 2, true);
  let o = 44;
  for (const c of chunks) { new Int16Array(buf, o, c.length).set(c); o += c.length * 2; }
  return new Blob([buf], { type: "audio/wav" });
}

// ------------------------------------------------------------------ tabs

function route() {
  const tab = (location.hash || "#live").slice(1);
  for (const sec of document.querySelectorAll(".tab")) sec.hidden = sec.id !== `tab-${tab}`;
  for (const a of document.querySelectorAll("nav a")) a.classList.toggle("active", a.dataset.tab === tab);
  status("");
  meters.forEach((m) => m.draw());
}

// ------------------------------------------------------------------ real-time

const live = { ws: null, mic: null, shown: 0, delays: [], models: [], timer: null };
const liveMeter = new Meter($("live-meter"));

function liveConnect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/realtime?model=${encodeURIComponent(model)}`);
  ws.binaryType = "arraybuffer";
  ws.onclose = () => {
    if (ws !== live.ws) return; // replaced after a model change
    liveStop();
    live.ws = null;
    setTimeout(liveConnect, 1500);
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "error") { status(msg.message, true); liveStop(); }
    else if (msg.type === "update") liveRender(msg);
  };
  live.ws = ws;
}

function liveRender({ phrases, pending, latency }) {
  const out = $("live-out");
  if (!live.shown && (phrases.length || pending)) out.querySelector(".empty")?.remove();
  out.querySelectorAll(".row.pending").forEach((r) => r.remove());
  for (const p of phrases.slice(live.shown)) {
    const r = row(p.start, p.text);
    r.classList.add("flash");
    out.append(r);
  }
  live.shown = phrases.length;
  if (pending) {
    const r = row(pending.start, pending.text);
    r.classList.add("pending");
    out.append(r);
  }
  out.scrollTop = out.scrollHeight;
  if (latency) {
    requestAnimationFrame(() => {
      live.delays.push(performance.now() - latency.stamp);
      live.models.push(latency.model_ms);
      live.delays = live.delays.slice(-20);
      live.models = live.models.slice(-20);
      $("live-delay").textContent = Math.round(median(live.delays));
      $("live-model").textContent = Math.round(median(live.models));
    });
  }
}

async function liveStart() {
  if (!live.ws || live.ws.readyState !== WebSocket.OPEN) return status("サーバーに接続できません", true);
  status("");
  try {
    live.mic = await openMic((pcm) => {
      // 8-byte timestamp (this page's clock) + PCM; the server echoes the newest stamp.
      if (live.ws?.readyState !== WebSocket.OPEN) return;
      const frame = new Uint8Array(8 + pcm.byteLength);
      new DataView(frame.buffer).setFloat64(0, performance.now(), true);
      frame.set(new Uint8Array(pcm), 8);
      live.ws.send(frame.buffer);
    }, liveMeter);
  } catch (err) {
    return status(micError(err), true);
  }
  live.ws.send(JSON.stringify({ type: "start" }));
  live.shown = 0; live.delays = []; live.models = [];
  $("live-out").replaceChildren();
  $("live-delay").textContent = $("live-model").textContent = "–";
  liveMeter.reset();
  live.timer = setInterval(() => { $("live-time").textContent = fmtClock(live.mic.elapsed()); }, 250);
  $("live-btn").classList.add("on");
  $("live-label").textContent = "停止";
}

function liveStop() {
  if (!live.mic) return;
  live.mic.close();
  live.mic = null;
  clearInterval(live.timer);
  if (live.ws?.readyState === WebSocket.OPEN) live.ws.send(JSON.stringify({ type: "stop" }));
  $("live-btn").classList.remove("on");
  $("live-label").textContent = "マイクで話す";
}

$("live-btn").onclick = () => (live.mic ? liveStop() : liveStart());
$("live-clear").onclick = () => {
  live.shown = 0;
  $("live-out").innerHTML = '<p class="empty">話すとここに文字起こしが表示されます。0.7 秒黙ると確定します。</p>';
};

// ------------------------------------------------------------------ record

const rec = { mic: null, chunks: [], timer: null, url: null };
const recMeter = new Meter($("rec-meter"));

async function recStart() {
  status("");
  rec.chunks = [];
  try {
    rec.mic = await openMic((pcm) => rec.chunks.push(new Int16Array(pcm)), recMeter);
  } catch (err) {
    return status(micError(err), true);
  }
  recMeter.reset();
  $("rec-audio").hidden = true;
  rec.timer = setInterval(() => { $("rec-time").textContent = fmtClock(rec.mic.elapsed()); }, 250);
  $("rec-btn").classList.add("on");
  $("rec-label").textContent = "停止して認識";
}

async function recStop() {
  rec.mic.close();
  rec.mic = null;
  clearInterval(rec.timer);
  $("rec-btn").classList.remove("on");
  $("rec-label").textContent = "録音開始";
  const blob = wav(rec.chunks);
  if (rec.url) URL.revokeObjectURL(rec.url);
  rec.url = URL.createObjectURL(blob);
  $("rec-audio").src = rec.url;
  $("rec-audio").hidden = false;
  $("rec-btn").disabled = true;
  status("認識中…");
  try {
    if ($("rec-compare").checked) await compareAll($("rec-out"), $("rec-stats"), blob);
    else showResult($("rec-out"), $("rec-stats"), await postAudio(blob));
    status("");
  } catch (err) {
    status(`認識に失敗しました: ${err.message}`, true);
  } finally {
    $("rec-btn").disabled = false;
  }
}

$("rec-btn").onclick = () => (rec.mic ? recStop() : recStart());

// ------------------------------------------------------------------ file

async function sendFile(file) {
  if (!file) return;
  status(`${file.name} を送信・認識中…`);
  $("file-out").replaceChildren();
  try {
    if ($("file-compare").checked) await compareAll($("file-out"), $("file-stats"), file);
    else showResult($("file-out"), $("file-stats"), await postAudio(file));
    status("");
  } catch (err) {
    status(`認識に失敗しました: ${err.message}`, true);
  }
}

const drop = $("drop");
$("file-input").onchange = (e) => { sendFile(e.target.files[0]); e.target.value = ""; };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); sendFile(e.dataTransfer.files[0]); };

// ------------------------------------------------------------------ start

const meters = [liveMeter, recMeter];
window.addEventListener("hashchange", () => { liveStop(); route(); });
window.addEventListener("resize", () => meters.forEach((m) => m.draw()));
$("model").onchange = (e) => {
  liveStop();
  setModel(e.target.value);
  const old = live.ws;
  live.ws = null;
  old?.close();
  liveConnect();
};

fetch("/api/info").then((r) => r.json()).then((info) => {
  models = info.models;
  $("model").replaceChildren(...models.map((m) => new Option(m.label, m.id)));
  const saved = savedModel();
  setModel(models.some((m) => m.id === saved) ? saved : models[0].id);
  $("model").value = model;
  liveConnect();
}).catch(() => { $("info").textContent = "サーバーに接続できません"; });
route();
