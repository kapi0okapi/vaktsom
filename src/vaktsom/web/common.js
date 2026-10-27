// Shared by the dashboard, the lifeguard post panel and the wearable view.

export const T = {
  kind: {
    bottom: "Ubevegelig person på bunnen",
    missing: "Person forsvunnet under vann",
    distress: "Mulig nødsituasjon i vannflaten",
    face_down: "Ubevegelig i vannflaten",
    test: "Testalarm",
    system: "Systemvarsel",
  },
  short: { bottom: "På bunnen", missing: "Forsvunnet", distress: "Nødsituasjon", face_down: "Ubevegelig", test: "Test", system: "System" },
  severity: { alarm: "Alarm", pre_alarm: "Forvarsel", system: "Systemvarsel" },
  outcome: { real: "Reell hendelse", false_alarm: "Falsk alarm", drill: "Øvelse", test: "Test", other: "Annet" },
  coverage: { ok: "Full dekning", reduced: "Én kameravinkel", degraded: "Redusert dekning", down: "Ingen dekning" },
  camera: { ok: "OK", degraded: "Redusert", down: "Ute", unknown: "Venter" },
  mode: { waves: "Bølger", bubbles: "Bobler" },
  reason: { glare: "gjenskinn", dark: "mørkt", blurred_or_fogged: "uskarpt eller dugg", frozen: "fryst bilde", low_fps: "lav bildefrekvens", no_frames: "ingen bilder" },
};

// Signs the page in. A panel or watch opened once with #token=... trades the token for a session
// that lasts until an administrator revokes the device; people without a session go to the login page.
export async function signIn() {
  const token = new URLSearchParams(location.hash.slice(1)).get("token");
  if (token) {
    await fetch("/api/device-session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }) });
    history.replaceState(null, "", location.pathname + location.search);
  }
  const r = await fetch("/api/me");
  if (r.status === 401) {
    location.href = `/login?next=${encodeURIComponent(location.pathname + location.search)}`;
    return new Promise(() => {});
  }
  return r.json();
}

export async function loadConfig() {
  const me = await signIn();
  const r = await fetch("/api/config");
  return { ...(await r.json()), me };
}

// The engine sends its state ten times a second. Silence for 5 s means the system is not watching,
// whether the server is gone or the connection is open but nothing arrives: onStatus(false) then.
export function connect(role, name, onMessage, onStatus) {
  let ws;
  let retry = 500;
  let last = 0, up = null;
  const status = (ok) => { if (ok !== up) { up = ok; onStatus?.(ok); } };
  const open = () => {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?role=${role}&name=${encodeURIComponent(name)}`);
    ws.onopen = () => { retry = 500; fetch("/api/me"); };  // keeps a device's session fresh
    ws.onmessage = (e) => { last = Date.now(); status(true); onMessage(JSON.parse(e.data)); };
    ws.onclose = (e) => {
      status(false);
      if (e.code === 4401) { location.reload(); return; }  // signed out or device revoked: sign in again
      setTimeout(open, retry); retry = Math.min(retry * 2, 5000);
    };
  };
  setInterval(() => { if (Date.now() - last > 5000) status(false); }, 1000);
  open();
  return { send: (msg) => ws && ws.readyState === 1 && ws.send(JSON.stringify(msg)) };
}

export const pad = (n) => String(n).padStart(2, "0");
export function clock(el) {
  const tick = () => { const d = new Date(); el.textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; };
  tick();
  setInterval(tick, 1000);
}
export function since(ts) {
  const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}
export const timeOf = (ts) => { const d = new Date(ts * 1000); return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; };

const NS = "http://www.w3.org/2000/svg";
export function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent?.appendChild(e);
  return e;
}

// Pool map in metres; y is flipped so north is up.
export class PoolMap {
  constructor(svg, cfg, { basins = null, cameras = true, margin = 3 } = {}) {
    this.svg = svg;
    this.cfg = cfg;
    this.only = basins;
    const shown = cfg.basins.filter((b) => !basins || basins.includes(b.id));
    const pts = shown.flatMap((b) => b.outline);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    this.box = { x0: Math.min(...xs) - margin, x1: Math.max(...xs) + margin, y0: Math.min(...ys) - margin, y1: Math.max(...ys) + margin };
    svg.setAttribute("viewBox", `${this.box.x0} ${-this.box.y1} ${this.box.x1 - this.box.x0} ${this.box.y1 - this.box.y0}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    this.static = el("g", {}, svg);
    this.cover = el("g", {}, svg);
    this.zones = el("g", {}, svg);
    this.people = el("g", {}, svg);
    this.marks = el("g", {}, svg);
    this.draw = el("g", {}, svg);
    el("rect", { x: this.box.x0, y: -this.box.y1, width: this.box.x1 - this.box.x0, height: this.box.y1 - this.box.y0, fill: "var(--deck)" }, this.static);
    for (const b of shown) {
      const poly = el("polygon", { points: b.outline.map(([x, y]) => `${x},${-y}`).join(" "), class: `basin${b.depth >= 3 ? " deep" : ""}`, "data-basin": b.id }, this.static);
      poly.dataset.basin = b.id;
      const [lx, ly] = [Math.min(...b.outline.map((p) => p[0])) + 0.6, Math.max(...b.outline.map((p) => p[1])) - 1.6];
      const label = el("text", { x: lx, y: -ly, class: "basin-label" }, this.static);
      label.textContent = b.name;
      const w = Math.max(...b.outline.map((p) => p[0])) - Math.min(...b.outline.map((p) => p[0]));
      const h = Math.max(...b.outline.map((p) => p[1])) - Math.min(...b.outline.map((p) => p[1]));
      if (w >= 20 && h >= 8) {
        const y0 = Math.min(...b.outline.map((p) => p[1]));
        const x0 = Math.min(...b.outline.map((p) => p[0]));
        const n = Math.floor(h / 2.5);
        for (let k = 1; k < n; k++) el("line", { x1: x0 + 0.5, x2: x0 + w - 0.5, y1: -(y0 + (k * h) / n), y2: -(y0 + (k * h) / n), class: "lane" }, this.static);
      }
    }
    if (cameras) {
      for (const c of cfg.cameras) {
        if (!c.position || (basins && !c.basins.some((b) => basins.includes(b)))) continue;
        const [x, y] = c.position, [tx, ty] = c.target;
        const a = Math.atan2(ty - y, tx - x);
        const s = 0.9;
        const tri = [[x + Math.cos(a) * s, y + Math.sin(a) * s], [x + Math.cos(a + 2.5) * s * 0.7, y + Math.sin(a + 2.5) * s * 0.7], [x + Math.cos(a - 2.5) * s * 0.7, y + Math.sin(a - 2.5) * s * 0.7]];
        el("polygon", { points: tri.map(([px, py]) => `${px},${-py}`).join(" "), class: "camicon" }, this.static).appendChild(document.createElementNS(NS, "title")).textContent = c.name || c.id;
      }
    }
  }

  toPool(evt) {
    const pt = this.svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const p = pt.matrixTransform(this.svg.getScreenCTM().inverse());
    return [p.x, -p.y];
  }

  update(state) {
    const alarmed = new Map(state.alarms.map((a) => [a.person_id, a]));
    this.people.replaceChildren();
    for (const p of state.persons) {
      if (this.only && !this.only.includes(p.basin)) continue;
      if (!p.basin) continue;
      let cls = "p";
      if (alarmed.has(p.id)) cls += " alarm";
      else if (p.lost) cls += " lost";
      else if (p.distress >= 0.8) cls += " distress";
      else if (p.under >= 0.6) cls += " under";
      el("circle", { cx: p.x, cy: -p.y, r: 0.42, class: cls }, this.people);
    }
    this.marks.replaceChildren();
    for (const a of state.alarms) {
      if (this.only && !this.only.includes(a.basin_id)) continue;
      const pre = a.severity === "pre_alarm";
      for (const delay of [0, 0.6]) {
        const r = el("circle", { cx: a.x, cy: -a.y, r: 1.2, class: `ring${pre ? " pre" : ""}` }, this.marks);
        r.style.animationDelay = `${delay}s`;
      }
      el("circle", { cx: a.x, cy: -a.y, r: 0.55, fill: pre ? "var(--warn)" : "var(--alarm)" }, this.marks);
    }
    this.cover.replaceChildren();
    for (const [bid, c] of Object.entries(state.coverage || {})) {
      if (this.only && !this.only.includes(bid)) continue;
      for (const [x, y] of c.uncovered_cells) el("rect", { x: x - 0.25, y: -y - 0.25, width: 0.5, height: 0.5, class: "uncovered" }, this.cover);
    }
    this.zones.replaceChildren();
    for (const z of state.zones || []) {
      el("polygon", { points: z.polygon.map(([x, y]) => `${x},${-y}`).join(" "), class: "zone" }, this.zones);
    }
    for (const poly of this.static.querySelectorAll("polygon.basin")) {
      poly.classList.toggle("closed", (state.closed || []).includes(poly.dataset.basin));
    }
  }
}

// A siren pattern that sounds like nothing else in a pool hall: two tones, fast alternation.
export class Siren {
  constructor() { this.ctx = null; this.timer = null; }
  unlock() { if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)(); return this.ctx.resume(); }
  beep(freq, dur, gain = 0.25) {
    if (!this.ctx) return;
    const o = this.ctx.createOscillator(), g = this.ctx.createGain();
    o.type = "square"; o.frequency.value = freq;
    g.gain.value = gain;
    o.connect(g).connect(this.ctx.destination);
    o.start(); o.stop(this.ctx.currentTime + dur);
  }
  start(pre = false) {
    if (this.timer) return;
    let k = 0;
    this.timer = setInterval(() => {
      if (pre) { if (k % 4 === 0) this.beep(880, 0.15, 0.12); }
      else this.beep(k % 2 ? 960 : 640, 0.22);
      k++;
    }, 250);
  }
  stop() { clearInterval(this.timer); this.timer = null; }
}
