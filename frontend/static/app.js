/* RehabSense frontend — vanilla JS single-page app.
   Screens: home -> exercise picker -> (live | upload) -> report.
   Live mode streams JPEG frames over /ws/live and draws the returned Kinect-25
   skeleton on an overlay canvas; upload mode POSTs the file and polls the job. */

"use strict";

const $ = (id) => document.getElementById(id);

const SCREENS = ["home", "exercise", "live", "upload", "report"];
const state = {
  mode: null,               // "live" | "upload"
  exercise: null,           // 1..5
  exercises: [],            // from /api/exercises
  bones: [],                // from /api/bones
  sessionSec: 20,
  ws: null,
  stream: null,
  pumping: false,
  recording: false,
  reportFrom: null,         // which mode produced the current report
  sid: null,                // server-side live-session id (survives reconnects)
  wantLive: false,          // we intend to hold a /ws/live connection
  reconnectAttempt: 0,
  pendingStop: false,       // session timer ended while disconnected -> stop on resume
};

/* ---------------------------------------------------------------- utils */
function show(name) {
  SCREENS.forEach((s) => { $("screen-" + s).hidden = s !== name; });
  window.scrollTo({ top: 0 });
}

let toastTimer = null;
function toast(msg, isErr = false) {
  const el = $("toast");
  el.textContent = msg;
  el.className = "toast" + (isErr ? " err" : "");
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 4200);
}

function bandColor(pct) {
  const s = getComputedStyle(document.documentElement);
  if (pct >= 80) return s.getPropertyValue("--good").trim();
  if (pct >= 55) return s.getPropertyValue("--warning").trim();
  return s.getPropertyValue("--critical").trim();
}
function verdictWord(pct) {
  return pct >= 88 ? "excellent" : pct >= 72 ? "good" : pct >= 55 ? "fair" : "needs work";
}

/* ---------------------------------------------------------------- boot */
async function boot() {
  try {
    const [ex, bones] = await Promise.all([
      fetch("/api/exercises").then((r) => r.json()),
      fetch("/api/bones").then((r) => r.json()),
    ]);
    state.exercises = ex;
    state.bones = bones;
  } catch {
    toast("Could not reach the server — is server.py running?", true);
  }
  pollEngine();
  buildExerciseGrid();
}

async function pollEngine() {
  const pill = $("engine-pill"), label = $("engine-label");
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    if (h.engine === "ready") {
      pill.className = "engine-pill ready"; label.textContent = "AI model ready";
      return;                                        // stop polling
    }
    if (h.engine === "error") {
      pill.className = "engine-pill error"; label.textContent = "AI model failed to load";
      return;
    }
    pill.className = "engine-pill"; label.textContent = "loading AI model…";
  } catch {
    pill.className = "engine-pill error"; label.textContent = "server offline";
  }
  setTimeout(pollEngine, 2000);
}

/* ------------------------------------------------------- exercise picker */
/* category chip per exercise: joint + movement type (clinical planes) */
const EX_TAGS = {
  1: "Trunk · side bend",
  2: "Trunk · forward bend",
  3: "Trunk · twist",
  4: "Hip · leg lift",
  5: "Hip · leg circles",
};

function buildExerciseGrid() {
  const grid = $("exercise-grid");
  grid.innerHTML = "";
  state.exercises.forEach((ex) => {
    const card = document.createElement("button");
    card.className = "ex-card";
    card.innerHTML = `<span class="ex-num">${ex.id}</span>
      <span class="ex-info"><span class="ex-tag">${EX_TAGS[ex.id] || ""}</span>
      <h3>${ex.name}</h3><p>${ex.howto}</p></span>
      <canvas class="ex-anim" width="336" height="260" data-ex="${ex.id}"
              aria-label="how to do ${ex.name}"></canvas>`;
    card.addEventListener("click", () => {
      state.exercise = ex.id;
      grid.querySelectorAll(".ex-card").forEach((c) => c.classList.remove("on"));
      card.classList.add("on");
      $("exercise-continue").disabled = false;
    });
    grid.appendChild(card);
  });
  startExAnims();
}

function openExercisePicker(mode) {
  state.mode = mode;
  $("exercise-title").textContent = "Choose your exercise";
  $("exercise-sub").textContent = mode === "live"
    ? "You will perform this in front of your camera."
    : "Pick the exercise shown in your video, so it is scored against the right movement profile.";
  $("exercise-continue").disabled = state.exercise === null;
  show("exercise");
}

function currentExercise() {
  return state.exercises.find((e) => e.id === state.exercise) || { name: "", howto: "" };
}

/* ------------------------------------------- exercise demo animations
   Each picker card carries a small canvas showing HOW to do the exercise.
   One shared requestAnimationFrame loop drives all of them; drawing is
   skipped while the picker screen is hidden. Figures are stick people:
   the static lower body in muted grey, the MOVING part in accent blue. */
const EX_COLORS = { static: "#898781", moving: "#3987e5", ground: "rgba(255,255,255,0.10)" };
let exAnimId = null;

function startExAnims() {
  if (exAnimId !== null) cancelAnimationFrame(exAnimId);
  const canvases = [...document.querySelectorAll(".ex-anim")];
  const t0 = performance.now();
  const loop = (now) => {
    exAnimId = requestAnimationFrame(loop);
    if ($("screen-exercise").hidden) return;               // nothing visible to draw
    const t = (now - t0) / 1000;
    for (const cv of canvases) drawExAnim(cv, +cv.dataset.ex, t);
  };
  exAnimId = requestAnimationFrame(loop);
}

function drawExAnim(cv, ex, t) {
  const ctx = cv.getContext("2d");
  // 336x260 canvas at 2x -> 168x130 units; scene is 150 wide, so centre it (+9)
  ctx.setTransform(2, 0, 0, 2, 18, 0);
  ctx.clearRect(-9, 0, 168, 130);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const fn = EX_ANIMS[ex] || drawNeutralFigure;
  fn(ctx, t);
}

/* rotate offset (x,y) by angle a and translate to point p */
function rot(p, x, y, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [p[0] + x * c - y * s, p[1] + x * s + y * c];
}

function easeInOutCubic(u) {
  return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
}

/* Keyframed motion: segments of {dur, from, to} (holds have from === to).
   Returns {v: value, vel: signed velocity} at time t, looping. */
function timeline(t, segs) {
  const total = segs.reduce((s, k) => s + k.dur, 0);
  let tt = t % total;
  for (const k of segs) {
    if (tt <= k.dur) {
      const u = easeInOutCubic(tt / k.dur);
      const eps = 0.02;
      const u2 = easeInOutCubic(Math.min(1, tt / k.dur + eps));
      return { v: k.from + (k.to - k.from) * u,
               vel: (k.to - k.from) * (u2 - u) / eps };
    }
    tt -= k.dur;
  }
  return { v: segs[0].from, vel: 0 };
}

/* ---------- shared figure primitives (150x130 unit space) ---------- */
const PELVIS = [75, 72], SPINE_LEN = 34, N_SEG = 6, HEAD_R = 7;

function drawGroundAndShadow(ctx) {
  ctx.fillStyle = "rgba(0,0,0,0.40)";                       // soft shadow under the feet
  ctx.beginPath(); ctx.ellipse(75, 119.5, 27, 3.4, 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = EX_COLORS.ground;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(24, 119); ctx.lineTo(126, 119); ctx.stroke();
}

function drawJoint(ctx, p, r = 2.1, col = "#6da7ec") {
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2); ctx.fill();
}

/* static jointed legs: hip -> knee -> ankle -> toes, plus the pelvis bar */
function drawLegs(ctx) {
  ctx.strokeStyle = EX_COLORS.static;
  ctx.lineWidth = 5;
  const hipL = [67, 72], hipR = [83, 72];
  const kneeL = [64, 94], kneeR = [86, 94];
  const ankL = [62, 117], ankR = [88, 117];
  ctx.beginPath();
  ctx.moveTo(hipL[0], hipL[1]); ctx.lineTo(kneeL[0], kneeL[1]); ctx.lineTo(ankL[0], ankL[1]);
  ctx.lineTo(ankL[0] - 7, 118);                                              // left foot
  ctx.moveTo(hipR[0], hipR[1]); ctx.lineTo(kneeR[0], kneeR[1]); ctx.lineTo(ankR[0], ankR[1]);
  ctx.lineTo(ankR[0] + 7, 118);                                              // right foot
  ctx.moveTo(hipL[0], hipL[1]); ctx.lineTo(hipR[0], hipR[1]);                // pelvis bar
  ctx.stroke();
  drawJoint(ctx, kneeL, 2, "#a5a39c"); drawJoint(ctx, kneeR, 2, "#a5a39c");
}

/* spine as a chain of vertebra segments, each adding a/N -> organic arc.
   Returns the polyline points + the tip direction angle. */
function spinePoints(a) {
  const pts = [PELVIS];
  let p = PELVIS, cum = 0;
  for (let i = 0; i < N_SEG; i++) {
    cum += a / N_SEG;
    p = rot(p, 0, -SPINE_LEN / N_SEG, cum);
    pts.push(p);
  }
  return { pts, tipAngle: cum };
}

/* upper body laterally bent by a. style: {color, width, alpha, joints} */
function drawBentTorso(ctx, a, style = {}) {
  const col = style.color || EX_COLORS.moving;
  const alpha = style.alpha ?? 1;
  ctx.save();
  ctx.globalAlpha = alpha;

  const { pts, tipAngle } = spinePoints(a);
  const top = pts[pts.length - 1];

  ctx.strokeStyle = col;
  ctx.lineWidth = style.width || 5;
  ctx.beginPath();                                           // vertebral spine
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const p of pts.slice(1)) ctx.lineTo(p[0], p[1]);
  ctx.stroke();

  const shL = rot(top, -10.5, 0, tipAngle);                  // shoulder girdle
  const shR = rot(top, 10.5, 0, tipAngle);
  ctx.beginPath(); ctx.moveTo(shL[0], shL[1]); ctx.lineTo(shR[0], shR[1]); ctx.stroke();

  // arms HANG under gravity (slight lag), with a soft elbow bow -> reads human.
  ctx.lineWidth = (style.width || 5) - 1;
  for (const [sh, side] of [[shL, -1], [shR, 1]]) {
    const hand = [sh[0] + side * 2 - Math.sin(a) * 5, sh[1] + 24];
    const elbow = [(sh[0] + hand[0]) / 2 + side * 3.5, (sh[1] + hand[1]) / 2];
    ctx.beginPath();
    ctx.moveTo(sh[0], sh[1]);
    ctx.quadraticCurveTo(elbow[0], elbow[1], hand[0], hand[1]);
    ctx.stroke();
  }

  const head = rot(top, 0, -(HEAD_R + 3.5), tipAngle * 1.12); // head leads slightly
  ctx.beginPath();                                            // neck
  ctx.moveTo(top[0], top[1]);
  const neckEnd = rot(top, 0, -3.5, tipAngle * 1.12);
  ctx.lineTo(neckEnd[0], neckEnd[1]);
  ctx.stroke();
  ctx.fillStyle = col;                                        // filled head
  ctx.beginPath(); ctx.arc(head[0], head[1], HEAD_R, 0, Math.PI * 2); ctx.fill();

  if (style.joints !== false) {
    drawJoint(ctx, shL); drawJoint(ctx, shR); drawJoint(ctx, PELVIS, 2.4);
  }
  ctx.restore();
}

/* range guide: dashed arc from minA to maxA about a pivot, ticks at the
   range ends (and 0), moving dot + arrowhead showing travel direction */
function drawRangeGuide(ctx, a, vel, minA, maxA, pivot = PELVIS, R = SPINE_LEN + HEAD_R * 2 + 9) {
  const toXY = (ang) => rot(pivot, 0, -R, ang);
  const pad = 0.06;
  ctx.save();
  ctx.strokeStyle = "rgba(195,194,183,0.28)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();                                           // arc (canvas angles from +x axis)
  ctx.arc(pivot[0], pivot[1], R, -Math.PI / 2 + minA - pad, -Math.PI / 2 + maxA + pad);
  ctx.stroke();
  ctx.setLineDash([]);
  for (const tick of new Set([minA, 0, maxA])) {             // ticks (dedupe 0)
    const p1 = rot(pivot, 0, -R + 3, tick), p2 = rot(pivot, 0, -R - 3, tick);
    ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
  }
  const dot = toXY(a);                                       // progress dot
  ctx.fillStyle = EX_COLORS.moving;
  ctx.beginPath(); ctx.arc(dot[0], dot[1], 3, 0, Math.PI * 2); ctx.fill();
  if (Math.abs(vel) > 0.12) {                                // direction arrowhead
    const dir = Math.sign(vel);
    const tip = toXY(a + dir * 0.16);
    const base1 = rot(toXY(a + dir * 0.05), 0, -3.2, a);
    const base2 = rot(toXY(a + dir * 0.05), 0, 3.2, a);
    ctx.beginPath();
    ctx.moveTo(tip[0], tip[1]); ctx.lineTo(base1[0], base1[1]);
    ctx.lineTo(base2[0], base2[1]); ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
}

/* live degree readout, top-right; lights up at the target range */
function drawDegrees(ctx, a, maxA) {
  const deg = Math.round(Math.abs(a) * 180 / Math.PI);
  const atMax = deg >= Math.round(maxA * 180 / Math.PI) - 2;
  ctx.font = "600 10px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.fillStyle = atMax ? EX_COLORS.moving : "rgba(137,135,129,0.9)";
  ctx.fillText(deg + "°", 144, 14);
}

/* --- exercise 1: Trunk Lateral Flexion --------------------------------
   Real exercise tempo: centre hold -> ease to 30° left -> hold at end
   range -> back to centre -> same to the right. Ghost poses mark the
   ±30° targets; legs and pelvis never move. */
const LAT_MAX = 30 * Math.PI / 180;
const LAT_KEYS = [
  { dur: 0.5, from: 0, to: 0 },
  { dur: 1.1, from: 0, to: -LAT_MAX },
  { dur: 0.6, from: -LAT_MAX, to: -LAT_MAX },
  { dur: 1.1, from: -LAT_MAX, to: 0 },
  { dur: 0.5, from: 0, to: 0 },
  { dur: 1.1, from: 0, to: LAT_MAX },
  { dur: 0.6, from: LAT_MAX, to: LAT_MAX },
  { dur: 1.1, from: LAT_MAX, to: 0 },
];

function drawLateralFlexion(ctx, t) {
  const { v: a, vel } = timeline(t, LAT_KEYS);
  drawGroundAndShadow(ctx);
  drawBentTorso(ctx, -LAT_MAX, { color: "#c3c2b7", width: 3, alpha: 0.14, joints: false });
  drawBentTorso(ctx, LAT_MAX, { color: "#c3c2b7", width: 3, alpha: 0.14, joints: false });
  drawLegs(ctx);
  drawRangeGuide(ctx, a, vel, -LAT_MAX, LAT_MAX);
  drawBentTorso(ctx, a);
  drawDegrees(ctx, a, LAT_MAX);
}

/* --- exercise 2: Trunk Forward Flexion (SIDE VIEW) --------------------
   The figure faces right; the spine ROUNDS forward (upper vertebrae bend
   more than lower -> chest curls toward the legs, like reaching for your
   toes), arms hang toward the floor. Target range 60°, matching the
   scoring profile. Legs and pelvis stay fixed. */
const FWD_HIP = [58, 72];
const FWD_MAX = 60 * Math.PI / 180;
// rounding: upper spine segments take progressively more of the bend
const FWD_W = (() => {
  const w = [0.55, 0.75, 0.95, 1.15, 1.3, 1.3];
  const s = w.reduce((x, y) => x + y, 0);
  return w.map((x) => x * N_SEG / s);
})();
const FWD_KEYS = [
  { dur: 0.55, from: 0, to: 0 },
  { dur: 1.3, from: 0, to: FWD_MAX },
  { dur: 0.7, from: FWD_MAX, to: FWD_MAX },
  { dur: 1.3, from: FWD_MAX, to: 0 },
];

/* side-view static legs: two legs slightly offset for depth, feet forward */
function drawSideLegs(ctx) {
  const leg = (dx, col) => {
    ctx.strokeStyle = col;
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(FWD_HIP[0] + dx, FWD_HIP[1]);
    ctx.lineTo(FWD_HIP[0] + dx + 3, 95);                     // knee, soft bend
    ctx.lineTo(FWD_HIP[0] + dx, 117);                        // ankle
    ctx.lineTo(FWD_HIP[0] + dx + 13, 118);                   // foot points forward
    ctx.stroke();
  };
  leg(4, "#6b6a65");                                         // far leg (darker)
  leg(0, EX_COLORS.static);
  drawJoint(ctx, [FWD_HIP[0] + 3, 95], 2, "#a5a39c");
}

/* side-view torso rounded forward by a. One shoulder (side view), arms hang. */
function drawSideTorso(ctx, a, style = {}) {
  const col = style.color || EX_COLORS.moving;
  ctx.save();
  ctx.globalAlpha = style.alpha ?? 1;
  ctx.strokeStyle = col;
  ctx.lineWidth = style.width || 5;

  const pts = [FWD_HIP];                                     // rounded spine chain
  let p = FWD_HIP, cum = 0;
  for (let i = 0; i < N_SEG; i++) {
    cum += (a / N_SEG) * FWD_W[i];
    p = rot(p, 0, -SPINE_LEN / N_SEG, cum);
    pts.push(p);
  }
  const top = pts[pts.length - 1], tipAngle = cum;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const q of pts.slice(1)) ctx.lineTo(q[0], q[1]);
  ctx.stroke();

  // arms hang from the shoulder toward the floor (both overlap in side view;
  // the far one is offset + darker for depth)
  const armLen = 25;
  const arm = (dx, c, wdt) => {
    ctx.strokeStyle = c;
    ctx.lineWidth = wdt;
    const hand = [top[0] + dx + Math.sin(a) * 3, top[1] + armLen];
    const elbow = [(top[0] + hand[0]) / 2 + 3, (top[1] + hand[1]) / 2];
    ctx.beginPath();
    ctx.moveTo(top[0] + dx, top[1]);
    ctx.quadraticCurveTo(elbow[0] + dx, elbow[1], hand[0], hand[1]);
    ctx.stroke();
  };
  if ((style.alpha ?? 1) === 1) arm(3, "#2a6ec6", (style.width || 5) - 1.5);
  arm(0, col, (style.width || 5) - 1);

  ctx.strokeStyle = col;
  ctx.lineWidth = style.width || 5;
  const headA = tipAngle * 1.15;                             // head continues the curl
  const neckEnd = rot(top, 0, -3.5, headA);
  ctx.beginPath(); ctx.moveTo(top[0], top[1]); ctx.lineTo(neckEnd[0], neckEnd[1]); ctx.stroke();
  const head = rot(top, 0, -(HEAD_R + 3.5), headA);
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.arc(head[0], head[1], HEAD_R, 0, Math.PI * 2); ctx.fill();
  // tiny nose tick so the facing direction reads at a glance
  const nc = Math.cos(headA), ns = Math.sin(headA);
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(head[0] + (HEAD_R - 1) * nc, head[1] + (HEAD_R - 1) * ns);
  ctx.lineTo(head[0] + (HEAD_R + 2.5) * nc, head[1] + (HEAD_R + 2.5) * ns);
  ctx.stroke();

  if (style.joints !== false) {
    drawJoint(ctx, top); drawJoint(ctx, FWD_HIP, 2.4);
  }
  ctx.restore();
}

function drawForwardFlexion(ctx, t) {
  const { v: a, vel } = timeline(t, FWD_KEYS);
  drawGroundAndShadow(ctx);
  drawSideTorso(ctx, FWD_MAX, { color: "#c3c2b7", width: 3, alpha: 0.14, joints: false });
  drawSideLegs(ctx);
  drawRangeGuide(ctx, a, vel, 0, FWD_MAX, FWD_HIP);
  drawSideTorso(ctx, a);
  drawDegrees(ctx, a, FWD_MAX);
}

/* placeholder for exercises whose animation isn't built yet: neutral pose */
function drawNeutralFigure(ctx) {
  drawGroundAndShadow(ctx);
  drawLegs(ctx);
  drawBentTorso(ctx, 0, { joints: false });
}

/* --- exercise 3: Trunk Rotation (FRONT VIEW + top-view compass) -------
   Feet and hips stay square to the camera; only the shoulder girdle
   twists. The twist is shown by foreshortening: the shoulder bar gets
   narrower as it rotates, the near shoulder is bright/thick and the far
   one dark/thin, and the head turns (nose tick slides). A top-view
   compass above the figure makes the twist angle explicit. */
const ROT_MAX = 35 * Math.PI / 180;                 // ±35° = the 70° scoring target
const ROT_KEYS = [                                   // slow, deliberate tempo (~10s cycle)
  { dur: 0.6, from: 0, to: 0 },
  { dur: 1.8, from: 0, to: -ROT_MAX },
  { dur: 0.8, from: -ROT_MAX, to: -ROT_MAX },
  { dur: 1.8, from: -ROT_MAX, to: 0 },
  { dur: 0.6, from: 0, to: 0 },
  { dur: 1.8, from: 0, to: ROT_MAX },
  { dur: 0.8, from: ROT_MAX, to: ROT_MAX },
  { dur: 1.8, from: ROT_MAX, to: 0 },
];

/* bird's-eye mini-view: what the twist looks like FROM ABOVE.
   Grey bar = hips (never move). Blue bar = shoulders, rotating over them.
   Small circle = head, with a nose dot showing where you're facing. */
function drawTwistCompass(ctx, yaw, vel, maxA) {
  const CX = 75, CY = 18, RX = 30, RY = 10;
  const pt = (ang) => [CX + RX * Math.sin(ang), CY + RY * Math.cos(ang)];
  const pad = 0.10;
  ctx.save();
  ctx.strokeStyle = "rgba(195,194,183,0.28)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();                                    // range arc (front of the ellipse)
  const a0 = -maxA - pad, a1 = maxA + pad;
  ctx.moveTo(...pt(a0));
  for (let u = 0; u <= 1.001; u += 0.05) ctx.lineTo(...pt(a0 + (a1 - a0) * u));
  ctx.stroke();
  ctx.setLineDash([]);
  for (const tick of new Set([-maxA, 0, maxA])) {     // ticks
    const [tx, ty] = pt(tick);
    const dx = (tx - CX) / RX, dy = (ty - CY) / RY;
    ctx.beginPath();
    ctx.moveTo(tx - dx * 2.5, ty - dy * 2.5);
    ctx.lineTo(tx + dx * 2.5, ty + dy * 2.5);
    ctx.stroke();
  }
  // hips seen from above: a static grey bar (they stay square)
  ctx.strokeStyle = EX_COLORS.static;
  ctx.lineWidth = 3.5;
  const hipL = pt(Math.PI / 2), hipR = pt(-Math.PI / 2);
  ctx.beginPath(); ctx.moveTo(hipL[0], hipL[1]); ctx.lineTo(hipR[0], hipR[1]); ctx.stroke();
  // shoulders seen from above: a blue bar rotating over the hips
  const shA = pt(yaw + Math.PI / 2), shB = pt(yaw - Math.PI / 2);
  ctx.strokeStyle = EX_COLORS.moving;
  ctx.lineWidth = 3.5;
  ctx.beginPath(); ctx.moveTo(shA[0], shA[1]); ctx.lineTo(shB[0], shB[1]); ctx.stroke();
  ctx.fillStyle = EX_COLORS.moving;
  for (const p of [shA, shB]) {
    ctx.beginPath(); ctx.arc(p[0], p[1], 2.2, 0, Math.PI * 2); ctx.fill();
  }
  // head from above + nose dot = facing direction
  ctx.beginPath(); ctx.arc(CX, CY, 3.6, 0, Math.PI * 2); ctx.fill();
  const nose = [CX + 6.5 * Math.sin(yaw), CY + 5.2 * Math.cos(yaw)];
  ctx.beginPath(); ctx.arc(nose[0], nose[1], 1.6, 0, Math.PI * 2); ctx.fill();
  if (Math.abs(vel) > 0.10) {                         // arrowhead along the arc
    const dir = Math.sign(vel);
    const tip = pt(yaw + dir * 0.22);
    const bse = pt(yaw + dir * 0.08);
    const tvx = tip[0] - bse[0], tvy = tip[1] - bse[1];
    const n = Math.hypot(tvx, tvy) || 1;
    const px = -tvy / n * 3, py = tvx / n * 3;
    ctx.beginPath();
    ctx.moveTo(tip[0], tip[1]);
    ctx.lineTo(bse[0] + px, bse[1] + py);
    ctx.lineTo(bse[0] - px, bse[1] - py);
    ctx.closePath(); ctx.fill();
  }
  ctx.restore();
}

/* circular "rotate" arrow around the torso while it twists */
function drawTwistArrow(ctx, yaw, vel) {
  if (Math.abs(vel) < 0.10) return;
  const CX = 75, CYE = 54, RX = 17, RY = 5;
  const pt = (q) => [CX + RX * Math.sin(q), CYE + RY * Math.cos(q)];
  const dir = Math.sign(vel);
  ctx.save();
  ctx.strokeStyle = "rgba(57,135,229,0.75)";
  ctx.fillStyle = "rgba(57,135,229,0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();                                    // front sweep of the ellipse
  ctx.moveTo(...pt(-dir * 1.0));
  for (let u = 0; u <= 1.001; u += 0.05) ctx.lineTo(...pt(-dir * 1.0 + dir * 2.0 * u));
  ctx.stroke();
  const tip = pt(dir * 1.25), bse = pt(dir * 0.95);   // arrowhead at the leading end
  const tvx = tip[0] - bse[0], tvy = tip[1] - bse[1];
  const n = Math.hypot(tvx, tvy) || 1;
  const px = -tvy / n * 3.2, py = tvx / n * 3.2;
  ctx.beginPath();
  ctx.moveTo(tip[0], tip[1]);
  ctx.lineTo(bse[0] + px, bse[1] + py);
  ctx.lineTo(bse[0] - px, bse[1] - py);
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

/* the twisting upper body; style.girdleOnly draws just shoulders+arms (ghost) */
function drawTwistFigure(ctx, yaw, style = {}) {
  const TOPY = 40, SH_W = 11;
  const top = [75, TOPY];
  const sinY = Math.sin(yaw);
  // exaggerated foreshortening so the twist reads at small size (bar stays
  // level -- a tilt reads as a side-bend, which is the wrong message)
  const half = SH_W * Math.cos(yaw * 1.6);
  const shL = [75 - half, TOPY], shR = [75 + half, TOPY];
  const nearIsR = sinY >= 0;                          // which shoulder is closer to camera
  ctx.save();
  ctx.globalAlpha = style.alpha ?? 1;

  const armLen = 24, depth = Math.abs(sinY);
  const drawArm = (sh, side, near) => {
    ctx.strokeStyle = near ? "#4b93ea" : "#2a6ec6";
    ctx.lineWidth = (near ? 4.2 : 3) + (near ? depth * 0.8 : -depth * 0.6);
    const hand = [sh[0] + side * 2, sh[1] + armLen];
    const elbow = [(sh[0] + hand[0]) / 2 + side * 3, (sh[1] + hand[1]) / 2];
    ctx.beginPath();
    ctx.moveTo(sh[0], sh[1]);
    ctx.quadraticCurveTo(elbow[0], elbow[1], hand[0], hand[1]);
    ctx.stroke();
  };
  const col = style.color || EX_COLORS.moving;
  const near = nearIsR ? shR : shL, far = nearIsR ? shL : shR;

  drawArm(far, nearIsR ? -1 : 1, false);                   // far arm behind
  if (!style.girdleOnly) {
    ctx.strokeStyle = col;                                 // vertical spine
    ctx.lineWidth = style.width || 5;
    ctx.beginPath(); ctx.moveTo(PELVIS[0], PELVIS[1]); ctx.lineTo(top[0], top[1]); ctx.stroke();
  }
  // shoulder bar in two halves: thick toward the camera, thin away from it
  ctx.strokeStyle = col;
  ctx.lineWidth = (style.width || 5) + depth * 1.6;
  ctx.beginPath(); ctx.moveTo(top[0], top[1]); ctx.lineTo(near[0], near[1]); ctx.stroke();
  ctx.lineWidth = Math.max(2.2, (style.width || 5) - depth * 2.2);
  ctx.beginPath(); ctx.moveTo(top[0], top[1]); ctx.lineTo(far[0], far[1]); ctx.stroke();
  drawArm(near, nearIsR ? 1 : -1, true);                   // near arm in front

  if (!style.girdleOnly) {
    const headYaw = yaw * 1.15;                            // head turns a bit further
    const headC = [75, TOPY - 3.5 - HEAD_R];
    ctx.beginPath();                                       // neck
    ctx.moveTo(top[0], top[1]); ctx.lineTo(headC[0], headC[1] + HEAD_R); ctx.stroke();
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(headC[0], headC[1], HEAD_R, 0, Math.PI * 2); ctx.fill();
    const nx = Math.sin(headYaw), ny = 0.22;               // nose tick slides as head turns
    const nn = Math.hypot(nx, ny);
    ctx.strokeStyle = col;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(headC[0] + (HEAD_R - 1) * nx / nn, headC[1] + (HEAD_R - 1) * ny / nn);
    ctx.lineTo(headC[0] + (HEAD_R + 2.5) * nx / nn, headC[1] + (HEAD_R + 2.5) * ny / nn);
    ctx.stroke();
  }
  if (style.joints !== false && !style.girdleOnly) {
    const rNear = 2.1 + depth * 1.0, rFar = Math.max(1.2, 2.1 - depth * 0.9);
    drawJoint(ctx, nearIsR ? shR : shL, rNear, "#7fb4f0");
    drawJoint(ctx, nearIsR ? shL : shR, rFar, "#2a6ec6");
    drawJoint(ctx, PELVIS, 2.4);
  }
  ctx.restore();
}

function drawTrunkRotation(ctx, t) {
  const { v: yaw, vel } = timeline(t, ROT_KEYS);
  drawGroundAndShadow(ctx);
  drawLegs(ctx);
  drawTwistCompass(ctx, yaw, vel, ROT_MAX);
  drawTwistFigure(ctx, yaw);
  drawTwistArrow(ctx, yaw, vel);
  drawDegrees(ctx, yaw, ROT_MAX);
}

/* --- exercise 4: Hip Abduction (front view) ---------------------------
   Standing side-leg raise: everything is static grey (standing leg,
   pelvis, trunk leaning slightly over the stance leg for balance, arms
   out) except the BLUE leg, which lifts sideways from straight-down to
   the 40° scoring target and lowers again. Pivot is the hip joint. */
const ABD_MAX = 40 * Math.PI / 180;
const ABD_HIP_L = [61, 72], ABD_HIP_R = [73, 72];
const ABD_LEG = 45;
const ABD_KEYS = [                                   // slow: lift 1.6s, hold, lower 1.6s
  { dur: 0.6, from: 0, to: 0 },
  { dur: 1.6, from: 0, to: ABD_MAX },
  { dur: 0.8, from: ABD_MAX, to: ABD_MAX },
  { dur: 1.6, from: ABD_MAX, to: 0 },
];

function drawAbdStatic(ctx) {
  ctx.strokeStyle = EX_COLORS.static;
  ctx.lineWidth = 5;
  // standing (left) leg, slightly under the body's weight
  ctx.beginPath();
  ctx.moveTo(ABD_HIP_L[0], ABD_HIP_L[1]);
  ctx.lineTo(58, 95); ctx.lineTo(57, 117); ctx.lineTo(49, 118);   // knee, ankle, foot
  ctx.moveTo(ABD_HIP_L[0], ABD_HIP_L[1]);                          // pelvis bar
  ctx.lineTo(ABD_HIP_R[0], ABD_HIP_R[1]);
  ctx.stroke();
  // trunk leans a touch over the stance leg (real balance strategy)
  const mid = [(ABD_HIP_L[0] + ABD_HIP_R[0]) / 2, 72];
  const top = [mid[0] - 3, 40];
  ctx.beginPath(); ctx.moveTo(mid[0], mid[1]); ctx.lineTo(top[0], top[1]); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(top[0] - 11, top[1]); ctx.lineTo(top[0] + 11, top[1]); ctx.stroke();
  ctx.lineWidth = 4;                                               // arms out for balance
  ctx.beginPath();
  ctx.moveTo(top[0] - 11, top[1]); ctx.lineTo(top[0] - 26, top[1] + 12);
  ctx.moveTo(top[0] + 11, top[1]); ctx.lineTo(top[0] + 26, top[1] + 12);
  ctx.stroke();
  ctx.fillStyle = EX_COLORS.static;                                // head (grey: not the mover)
  ctx.beginPath(); ctx.arc(top[0], top[1] - 3.5 - HEAD_R, HEAD_R, 0, Math.PI * 2); ctx.fill();
  drawJoint(ctx, [58, 95], 2, "#a5a39c");
}

/* the abducting leg at angle th from straight-down (th >= 0 lifts to the right) */
function drawAbdLeg(ctx, th, style = {}) {
  const col = style.color || EX_COLORS.moving;
  const dirX = Math.sin(th), dirY = Math.cos(th);
  const knee = [ABD_HIP_R[0] + ABD_LEG * 0.52 * dirX, ABD_HIP_R[1] + ABD_LEG * 0.52 * dirY];
  const ankle = [ABD_HIP_R[0] + ABD_LEG * dirX, ABD_HIP_R[1] + ABD_LEG * dirY];
  ctx.save();
  ctx.globalAlpha = style.alpha ?? 1;
  ctx.strokeStyle = col;
  ctx.lineWidth = style.width || 5;
  ctx.beginPath();
  ctx.moveTo(ABD_HIP_R[0], ABD_HIP_R[1]);
  ctx.lineTo(knee[0], knee[1]);
  ctx.lineTo(ankle[0], ankle[1]);
  ctx.lineTo(ankle[0] + 7 * dirY, ankle[1] - 7 * dirX);            // foot, perpendicular
  ctx.stroke();
  if (style.joints !== false) {
    drawJoint(ctx, ABD_HIP_R, 2.6);                                // the pivot
    drawJoint(ctx, knee, 2, "#6da7ec");
  }
  ctx.restore();
}

/* downward fan guide at the hip: dashed arc from straight-down to the target */
function drawAbdGuide(ctx, th, vel, maxA) {
  const R = ABD_LEG + 7;
  const pt = (ang) => [ABD_HIP_R[0] + R * Math.sin(ang), ABD_HIP_R[1] + R * Math.cos(ang)];
  const pad = 0.06;
  ctx.save();
  ctx.strokeStyle = "rgba(195,194,183,0.28)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(...pt(-pad));
  for (let u = 0; u <= 1.001; u += 0.05) ctx.lineTo(...pt(-pad + (maxA + 2 * pad) * u));
  ctx.stroke();
  ctx.setLineDash([]);
  for (const tick of [0, maxA]) {                                  // radial ticks
    const dirX = Math.sin(tick), dirY = Math.cos(tick);
    ctx.beginPath();
    ctx.moveTo(ABD_HIP_R[0] + (R - 3) * dirX, ABD_HIP_R[1] + (R - 3) * dirY);
    ctx.lineTo(ABD_HIP_R[0] + (R + 3) * dirX, ABD_HIP_R[1] + (R + 3) * dirY);
    ctx.stroke();
  }
  const dot = pt(th);
  ctx.fillStyle = EX_COLORS.moving;
  ctx.beginPath(); ctx.arc(dot[0], dot[1], 3, 0, Math.PI * 2); ctx.fill();
  if (Math.abs(vel) > 0.10) {                                      // arrowhead, tangent
    const dir = Math.sign(vel);
    const tip = pt(th + dir * 0.14), bse = pt(th + dir * 0.04);
    const tvx = tip[0] - bse[0], tvy = tip[1] - bse[1];
    const n = Math.hypot(tvx, tvy) || 1;
    const px = -tvy / n * 3, py = tvx / n * 3;
    ctx.beginPath();
    ctx.moveTo(tip[0], tip[1]);
    ctx.lineTo(bse[0] + px, bse[1] + py);
    ctx.lineTo(bse[0] - px, bse[1] - py);
    ctx.closePath(); ctx.fill();
  }
  ctx.restore();
}

function drawHipAbduction(ctx, t) {
  const { v: th, vel } = timeline(t, ABD_KEYS);
  drawGroundAndShadow(ctx);
  drawAbdLeg(ctx, ABD_MAX, { color: "#c3c2b7", width: 3, alpha: 0.14, joints: false });
  drawAbdStatic(ctx);
  drawAbdGuide(ctx, th, vel, ABD_MAX);
  drawAbdLeg(ctx, th);
  drawDegrees(ctx, th, ABD_MAX);
}

/* --- exercise 5: Hip Circumduction (front view) -----------------------
   The whole leg sweeps a continuous circle from the ball-and-socket hip:
   forward -> out -> back -> centre in one smooth motion. The ankle traces
   an ellipse (a circle seen from the front); depth is faked by making the
   leg thicker/brighter when the foot passes the near side. A dashed
   ellipse shows the target path, a fading trail follows the ankle, and
   the readout counts the sweep 0-360°. Standing body reused from ex 4. */
const CIR_PERIOD = 4.5;                              // seconds per full circle
const CIR_CX = ABD_HIP_R[0] + 8, CIR_CY = ABD_HIP_R[1] + 40;
const CIR_RX = 16, CIR_RY = 6;

function cirAnkle(phi) {
  return [CIR_CX + CIR_RX * Math.sin(phi), CIR_CY - CIR_RY * Math.cos(phi)];
}

function drawCirPath(ctx) {
  ctx.save();
  ctx.strokeStyle = "rgba(195,194,183,0.28)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(...cirAnkle(0));
  for (let u = 0; u <= 1.001; u += 0.04) ctx.lineTo(...cirAnkle(u * 2 * Math.PI));
  ctx.stroke();
  ctx.restore();
}

function drawCirLeg(ctx, phi) {
  const ank = cirAnkle(phi);
  const front = -Math.cos(phi);                      // +1 when the foot is nearest the viewer
  const knee = [ABD_HIP_R[0] + (ank[0] - ABD_HIP_R[0]) * 0.55,
                ABD_HIP_R[1] + (ank[1] - ABD_HIP_R[1]) * 0.55];
  ctx.save();
  ctx.strokeStyle = front > 0 ? "#4b93ea" : EX_COLORS.moving;
  ctx.lineWidth = 5 + front * 1.2;                   // nearer = thicker
  ctx.beginPath();
  ctx.moveTo(ABD_HIP_R[0], ABD_HIP_R[1]);
  ctx.lineTo(knee[0], knee[1]);
  ctx.lineTo(ank[0], ank[1]);
  ctx.stroke();
  const legX = ank[0] - ABD_HIP_R[0], legY = ank[1] - ABD_HIP_R[1];
  const n = Math.hypot(legX, legY) || 1;
  ctx.beginPath();                                   // foot, perpendicular to the leg
  ctx.moveTo(ank[0], ank[1]);
  ctx.lineTo(ank[0] + 7 * legY / n, ank[1] - 7 * legX / n);
  ctx.stroke();
  drawJoint(ctx, ABD_HIP_R, 2.6);                    // the ball-and-socket pivot
  drawJoint(ctx, knee, 2, "#6da7ec");
  ctx.restore();
}

function drawCirTrail(ctx, phi) {
  ctx.save();
  ctx.lineCap = "round";
  for (let i = 1; i <= 9; i++) {                     // fading tail behind the ankle
    const a = phi - i * 0.13, b = phi - (i - 1) * 0.13;
    ctx.strokeStyle = `rgba(57,135,229,${0.34 * (1 - i / 10)})`;
    ctx.lineWidth = 3.2 * (1 - i / 12);
    ctx.beginPath();
    ctx.moveTo(...cirAnkle(a)); ctx.lineTo(...cirAnkle(b));
    ctx.stroke();
  }
  const tip = cirAnkle(phi + 0.30), bse = cirAnkle(phi + 0.14);  // direction arrow ahead
  const tvx = tip[0] - bse[0], tvy = tip[1] - bse[1];
  const n = Math.hypot(tvx, tvy) || 1;
  const px = -tvy / n * 3, py = tvx / n * 3;
  ctx.fillStyle = "rgba(57,135,229,0.85)";
  ctx.beginPath();
  ctx.moveTo(tip[0], tip[1]);
  ctx.lineTo(bse[0] + px, bse[1] + py);
  ctx.lineTo(bse[0] - px, bse[1] - py);
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawHipCircumduction(ctx, t) {
  const phi = (2 * Math.PI * t) / CIR_PERIOD;
  drawGroundAndShadow(ctx);
  drawAbdStatic(ctx);
  drawCirPath(ctx);
  drawCirTrail(ctx, phi);
  drawCirLeg(ctx, phi);
  const deg = Math.round((phi * 180 / Math.PI) % 360);
  ctx.font = "600 10px system-ui, sans-serif";       // sweep counter, not a target
  ctx.textAlign = "right";
  ctx.fillStyle = "rgba(137,135,129,0.9)";
  ctx.fillText(deg + "°", 144, 14);
}

const EX_ANIMS = {
  1: drawLateralFlexion,
  2: drawForwardFlexion,
  3: drawTrunkRotation,
  4: drawHipAbduction,
  5: drawHipCircumduction,
};

/* ============================================================ LIVE MODE */
const cam = $("cam"), overlay = $("overlay");
const veil = $("stage-veil"), veilContent = $("veil-content");
const trackChip = $("track-chip"), trackLabel = $("track-label");

let sendCanvas = null;      // offscreen canvas for JPEG capture
let pumpWatchdog = null;
let lastSend = 0;
let sessionTimer = null;
let sessionEndsAt = 0;
let reconnectTimer = null;
let hbTimer = null;             // heartbeat ping
let rxWatchdog = null;          // half-open detector
let lastRx = 0;                 // performance.now() of last message from server

async function enterLive() {
  const ex = currentExercise();
  $("live-ex-name").textContent = ex.name;
  $("live-ex-howto").textContent = ex.howto;
  resetLivePanel();
  state.sid = null;
  state.pendingStop = false;
  state.reconnectAttempt = 0;
  show("live");
  veil.hidden = false;
  veilContent.innerHTML = "<p>Waiting for camera…</p>";
  trackChip.className = "track-chip"; trackLabel.textContent = "connecting…";

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  } catch (e) {
    veilContent.innerHTML = `<p class="veil-err">Camera access was blocked.<br>
      Allow camera permission for this page (or close other apps using the camera),
      then press Back and try again.</p>`;
    return;
  }
  cam.srcObject = state.stream;
  await new Promise((res) => { cam.onloadedmetadata = res; });
  state.wantLive = true;
  startHeartbeat();
  connectLiveWS();
}

function connectLiveWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.onopen = () => {
    state.reconnectAttempt = 0;
    lastRx = performance.now();
    trackLabel.textContent = state.sid ? "reconnecting session…" : "starting tracker…";
  };
  ws.onmessage = (ev) => { lastRx = performance.now(); handleLiveMsg(JSON.parse(ev.data)); };
  ws.onclose = () => onWSDown();
  ws.onerror = () => { try { ws.close(); } catch {} };   // -> onclose -> reconnect
}

/* connection dropped (or we forced it closed): pause the pump and retry with backoff */
function onWSDown() {
  if (!state.wantLive) return;                            // intentional close (leaveLive)
  state.pumping = false;
  trackChip.className = "track-chip off";
  const n = ++state.reconnectAttempt;
  trackLabel.textContent = `reconnecting… (${n})`;
  if (state.recording || state.pendingStop) {
    veil.hidden = false;
    veilContent.innerHTML = "<p>Connection lost — reconnecting to save your session…</p>";
  }
  const delay = Math.min(6000, 400 * Math.pow(2, n - 1));  // 0.4s,0.8s,1.6s,…,6s
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => { if (state.wantLive) connectLiveWS(); }, delay);
}

function startHeartbeat() {
  clearInterval(hbTimer);
  clearInterval(rxWatchdog);
  hbTimer = setInterval(() => {
    if (state.ws && state.ws.readyState === 1) {
      try { state.ws.send(JSON.stringify({ type: "ping" })); } catch {}
    }
  }, 2000);
  rxWatchdog = setInterval(() => {
    if (!state.wantLive) return;
    // open socket but no traffic for 5s => half-open; force a clean reconnect
    if (state.ws && state.ws.readyState === 1 && performance.now() - lastRx > 5000) {
      try { state.ws.close(); } catch {}
    }
  }, 1000);
}

function stopHeartbeat() {
  clearInterval(hbTimer); hbTimer = null;
  clearInterval(rxWatchdog); rxWatchdog = null;
}

function handleLiveMsg(m) {
  if (m.type === "ready") {
    veil.hidden = !state.sid;                    // keep veil up while a resume is pending
    state.pumping = true;
    if (state.sid) {                             // reconnected mid-session -> resume it
      sendCmd({ type: "resume", sid: state.sid });
    }
    pumpFrame();
  } else if (m.type === "pose") {
    if (m.detected) {
      trackChip.className = "track-chip on"; trackLabel.textContent = "tracking";
      drawSkeleton(m.pts);
    } else {
      trackChip.className = "track-chip off";
      trackLabel.textContent = "no person — step back, full body in frame";
      clearOverlay();
    }
    scheduleNextFrame();
  } else if (m.type === "recording") {
    state.sid = m.sid;                            // remember the session for resume
  } else if (m.type === "resumed") {
    veil.hidden = true;
    trackChip.className = "track-chip on"; trackLabel.textContent = "tracking";
    if (state.pendingStop) {                      // timer had ended while offline
      state.pendingStop = false;
      sendCmd({ type: "stop" });
    }
  } else if (m.type === "resume_failed") {
    // session expired past the resume window -> can't recover it
    state.recording = false; state.sid = null; state.pendingStop = false;
    stopSessionTimer();
    resetLivePanel();
    veil.hidden = true;
    toast("Connection was lost too long to recover the session. Please start again.", true);
  } else if (m.type === "pong") {
    // liveness only; lastRx already updated by onmessage
  } else if (m.type === "live") {
    setLiveScore(m.score, m.reps);
  } else if (m.type === "analyzing") {
    stopSessionTimer();
    veil.hidden = false;
    veilContent.innerHTML = "<p>Analyzing your session…</p>";
  } else if (m.type === "report") {
    state.pumping = false;
    state.sid = null;
    leaveLive(false);
    renderReport(m.report, "live");
  } else if (m.type === "error") {
    state.recording = false; state.sid = null; state.pendingStop = false;
    stopSessionTimer();
    resetLivePanel();
    veil.hidden = true;
    toast(m.msg, true);
  }
}

function sendCmd(obj) {
  if (state.ws && state.ws.readyState === 1) {
    try { state.ws.send(JSON.stringify(obj)); return true; } catch {}
  }
  return false;
}

/* --- frame pump: ack-paced (next frame after the server answers), ~15 fps cap */
function pumpFrame() {
  if (!state.pumping || !state.ws || state.ws.readyState !== 1) return;
  if (!sendCanvas) sendCanvas = document.createElement("canvas");
  const vw = cam.videoWidth, vh = cam.videoHeight;
  if (!vw) { scheduleNextFrame(); return; }
  const w = Math.min(640, vw), h = Math.round(vh * (w / vw));
  sendCanvas.width = w; sendCanvas.height = h;
  sendCanvas.getContext("2d").drawImage(cam, 0, 0, w, h);
  sendCanvas.toBlob(async (blob) => {
    if (!blob || !state.pumping || state.ws?.readyState !== 1) return;
    lastSend = performance.now();
    state.ws.send(await blob.arrayBuffer());
    // watchdog: if no reply (message lost / hiccup), resume anyway
    clearTimeout(pumpWatchdog);
    pumpWatchdog = setTimeout(() => { if (state.pumping) pumpFrame(); }, 900);
  }, "image/jpeg", 0.7);
}

function scheduleNextFrame() {
  if (!state.pumping) return;
  clearTimeout(pumpWatchdog);
  const wait = Math.max(0, 66 - (performance.now() - lastSend));
  pumpWatchdog = setTimeout(pumpFrame, wait);
}

/* --- skeleton overlay (video uses object-fit: cover; both are CSS-mirrored) */
function drawSkeleton(pts) {
  const ctx = overlay.getContext("2d");
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  if (overlay.width !== cw || overlay.height !== ch) { overlay.width = cw; overlay.height = ch; }
  ctx.clearRect(0, 0, cw, ch);
  const vw = cam.videoWidth, vh = cam.videoHeight;
  if (!vw) return;
  const s = Math.max(cw / vw, ch / vh);                  // cover scale
  const ox = (cw - vw * s) / 2, oy = (ch - vh * s) / 2;
  const P = pts.map(([x, y]) => [ox + x * vw * s, oy + y * vh * s]);

  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(57, 135, 229, 0.9)";
  ctx.beginPath();
  for (const [a, b] of state.bones) {
    ctx.moveTo(P[a][0], P[a][1]);
    ctx.lineTo(P[b][0], P[b][1]);
  }
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  for (let i = 0; i < 20; i++) {                          // skip hand-tip detail joints
    ctx.beginPath();
    ctx.arc(P[i][0], P[i][1], 3.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function clearOverlay() {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

/* --- session control */
function startSession() {
  if (!state.ws || state.ws.readyState !== 1) { toast("Not connected yet — one moment.", true); return; }
  let n = 3;
  $("countdown").hidden = false;
  $("countdown-num").textContent = n;
  const tick = setInterval(() => {
    n -= 1;
    if (n <= 0) {
      clearInterval(tick);
      $("countdown").hidden = true;
      beginRecording();
    } else {
      $("countdown-num").textContent = n;
    }
  }, 900);
}

function beginRecording() {
  state.recording = true;
  state.pendingStop = false;
  sendCmd({ type: "start", exercise: state.exercise });
  $("live-start").hidden = true;
  $("live-stop").hidden = false;
  $("rec-chip").hidden = false;
  sessionEndsAt = performance.now() + state.sessionSec * 1000;
  sessionTimer = setInterval(() => {
    const left = Math.max(0, (sessionEndsAt - performance.now()) / 1000);
    $("live-remaining").textContent = left.toFixed(0);
    $("rec-time").textContent = (state.sessionSec - left).toFixed(1) + "s";
    if (left <= 0) endSession();
  }, 100);
}

function endSession() {
  if (!state.recording) return;
  state.recording = false;
  stopSessionTimer();
  if (!sendCmd({ type: "stop" })) {
    // disconnected right as the session ended -> stop once we reconnect + resume
    state.pendingStop = true;
    veil.hidden = false;
    veilContent.innerHTML = "<p>Reconnecting to finish your session…</p>";
  }
}

function stopSessionTimer() {
  clearInterval(sessionTimer);
  sessionTimer = null;
  $("rec-chip").hidden = true;
}

function setLiveScore(score, reps) {
  $("live-score").textContent = Math.round(score);
  $("live-reps").textContent = reps;
  const fill = $("live-meter");
  fill.style.width = Math.max(2, Math.min(100, score)) + "%";
  fill.style.background = bandColor(score);
}

function resetLivePanel() {
  state.recording = false;
  stopSessionTimer();
  $("live-score").textContent = "–";
  $("live-reps").textContent = "0";
  $("live-remaining").textContent = state.sessionSec;
  $("live-meter").style.width = "0%";
  $("live-start").hidden = false;
  $("live-stop").hidden = true;
}

function leaveLive(showHome = true) {
  state.wantLive = false;              // stop any reconnect attempts first
  state.pumping = false;
  state.recording = false;
  state.pendingStop = false;
  stopSessionTimer();
  stopHeartbeat();
  clearTimeout(pumpWatchdog);
  clearTimeout(reconnectTimer);
  if (state.ws) {
    if (state.sid) { try { state.ws.send(JSON.stringify({ type: "abort" })); } catch {} }
    try { state.ws.close(); } catch {}
    state.ws = null;
  }
  state.sid = null;
  if (state.stream) { state.stream.getTracks().forEach((t) => t.stop()); state.stream = null; }
  cam.srcObject = null;
  clearOverlay();
  $("countdown").hidden = true;
  if (showHome) show("exercise");
}

/* ============================================================ UPLOAD MODE */
let pickedFile = null;
let pollTimer = null;

function enterUpload() {
  const ex = currentExercise();
  $("upload-ex-name").textContent = "Upload — " + ex.name;
  $("upload-ex-howto").textContent = "The video should show: " + ex.howto.toLowerCase();
  clearPicked();
  $("progress-wrap").hidden = true;
  show("upload");
}

function setPicked(file) {
  const ok = /\.(mp4|mov|avi|webm|mkv|m4v)$/i.test(file.name);
  if (!ok) { toast("Unsupported file type — use MP4, MOV, AVI, WEBM or MKV.", true); return; }
  pickedFile = file;
  $("drop-idle").hidden = true;
  $("drop-picked").hidden = false;
  $("picked-name").textContent = file.name;
  $("picked-size").textContent = (file.size / 1048576).toFixed(1) + " MB";
  const prev = $("upload-preview");
  prev.src = URL.createObjectURL(file);
  prev.play().catch(() => {});
  $("upload-analyze").disabled = false;
}

function clearPicked() {
  pickedFile = null;
  $("drop-idle").hidden = false;
  $("drop-picked").hidden = true;
  const prev = $("upload-preview");
  if (prev.src) { URL.revokeObjectURL(prev.src); prev.removeAttribute("src"); }
  $("file-input").value = "";
  $("upload-analyze").disabled = true;
}

async function analyzeUpload() {
  if (!pickedFile) return;
  $("upload-analyze").disabled = true;
  $("progress-wrap").hidden = false;
  setProgress(0, "uploading video…");
  const fd = new FormData();
  fd.append("file", pickedFile);
  fd.append("exercise", state.exercise);
  let jobId;
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    jobId = (await res.json()).job_id;
  } catch (e) {
    toast("Upload failed: " + e.message, true);
    $("upload-analyze").disabled = false;
    $("progress-wrap").hidden = true;
    return;
  }
  pollJob(jobId);
}

function pollJob(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let job;
    try {
      job = await fetch("/api/job/" + jobId).then((r) => r.json());
    } catch { return; }
    if (job.state === "processing") {
      setProgress(job.progress, job.stage);
    } else if (job.state === "done") {
      clearInterval(pollTimer);
      setProgress(1, "done");
      renderReport(job.report, "upload");
    } else if (job.state === "error") {
      clearInterval(pollTimer);
      toast("Analysis failed: " + job.error, true);
      $("upload-analyze").disabled = false;
      $("progress-wrap").hidden = true;
    }
  }, 600);
}

function setProgress(frac, stage) {
  $("progress-fill").style.width = Math.round(frac * 100) + "%";
  $("progress-pct").textContent = Math.round(frac * 100) + "%";
  $("progress-stage").textContent = stage;
}

/* ============================================================ REPORT */
function starSVG(filled) {
  const p = [];
  for (let i = 0; i < 10; i++) {
    const ang = -Math.PI / 2 + (i * Math.PI) / 5;
    const r = i % 2 === 0 ? 12 : 5;
    p.push(`${13 + r * Math.cos(ang)},${13 + r * Math.sin(ang)}`);
  }
  return `<svg viewBox="0 0 26 26"><polygon points="${p.join(" ")}"
      class="${filled ? "fill" : "empty"}"/></svg>`;
}

function prettyDetail(detail) {
  const nice = {
    excursion_deg: "excursion", target_deg: "target", sparc: "SPARC",
    left_deg: "left", right_deg: "right", pos_deg: "one way", neg_deg: "other way",
    tilt_std_deg: "shoulder tilt σ", reps: "reps", cadence_per_min: "cadence/min",
    rhythm: "rhythm", rep_cv: "rep variability", note: "",
  };
  return Object.entries(detail || {})
    .map(([k, v]) => (nice[k] === "" ? v : `${nice[k] ?? k} ${v}${k.endsWith("_deg") ? "°" : ""}`))
    .join("  ·  ");
}

function renderReport(rep, from) {
  state.reportFrom = from;
  $("rep-exercise").textContent = rep.exercise_name;
  const when = new Date().toLocaleString(undefined,
    { dateStyle: "medium", timeStyle: "short" });
  $("rep-meta").textContent =
    (from === "live" ? "Live session" : "Video analysis") + "  ·  " + when;

  const comp = rep.composite;
  const col = comp >= 70 ? bandColor(85) : comp >= 50 ? bandColor(60) : bandColor(30);
  const band = $("rep-band");
  band.textContent = rep.band;
  band.style.color = col;

  // score ring (animate after layout)
  $("rep-score").textContent = Math.round(comp);
  const ring = $("ring-fill");
  const C = 540.35;
  ring.style.stroke = col;
  ring.style.strokeDashoffset = C;
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      ring.style.strokeDashoffset = C * (1 - Math.min(comp, 100) / 100);
    }));

  $("rep-stars").innerHTML =
    [1, 2, 3, 4, 5].map((i) => starSVG(i <= rep.stars)).join("");

  const ai = $("rep-ai");
  if (rep.ai_badge != null) {
    ai.hidden = false;
    ai.textContent = `AI model score  ${Math.round(rep.ai_badge)} / 50`;
  } else { ai.hidden = true; }
  $("rep-reps").textContent = rep.reps + (rep.reps === 1 ? " rep" : " reps");
  $("rep-dur").textContent = rep.duration_s.toFixed(0) + "s";

  const rows = $("rep-metrics");
  rows.innerHTML = "";
  rep.metrics.forEach((m) => {
    const c = bandColor(m.pct);
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `
      <span class="m-label">${m.label}</span>
      <div class="meter"><div class="meter-track">
        <div class="meter-fill" style="width:0%; background:${c}"></div>
      </div></div>
      <span class="m-pct" style="color:${c}">${Math.round(m.pct)}%</span>
      <span class="m-verdict">${verdictWord(m.pct)}</span>
      <span class="m-detail">${prettyDetail(m.detail)}</span>`;
    rows.appendChild(row);
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        row.querySelector(".meter-fill").style.width =
          Math.max(2, Math.min(100, m.pct)) + "%";
      }));
  });

  const fill = (ul, items) => {
    ul.innerHTML = "";
    items.forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s;
      ul.appendChild(li);
    });
  };
  fill($("rep-strengths"), rep.strengths);
  fill($("rep-improvements"), rep.improvements);

  const png = $("rep-png");
  if (rep.png) { png.hidden = false; png.href = rep.png; } else { png.hidden = true; }

  show("report");
}

/* ============================================================ wiring */
$("brand-home").addEventListener("click", () => { leaveLive(false); show("home"); });

document.querySelectorAll(".mode-card").forEach((c) =>
  c.addEventListener("click", () => openExercisePicker(c.dataset.mode)));

document.querySelectorAll("[data-back]").forEach((b) =>
  b.addEventListener("click", () => show(b.dataset.back)));

$("exercise-continue").addEventListener("click", () => {
  if (state.exercise == null) return;
  state.mode === "live" ? enterLive() : enterUpload();
});

$("live-back").addEventListener("click", () => leaveLive(true));
$("live-start").addEventListener("click", startSession);
$("live-stop").addEventListener("click", endSession);

$("seg-length").querySelectorAll("button").forEach((b) =>
  b.addEventListener("click", () => {
    $("seg-length").querySelectorAll("button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    state.sessionSec = parseInt(b.dataset.sec, 10);
    if (!state.recording) $("live-remaining").textContent = state.sessionSec;
  }));

$("upload-back").addEventListener("click", () => { clearInterval(pollTimer); show("exercise"); });

const dz = $("dropzone");
dz.addEventListener("click", (e) => {
  if (e.target.id !== "picked-clear") $("file-input").click();
});
dz.addEventListener("keydown", (e) => { if (e.key === "Enter") $("file-input").click(); });
["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) setPicked(e.dataTransfer.files[0]);
});
$("file-input").addEventListener("change", (e) => {
  if (e.target.files.length) setPicked(e.target.files[0]);
});
$("picked-clear").addEventListener("click", (e) => { e.stopPropagation(); clearPicked(); });
$("upload-analyze").addEventListener("click", analyzeUpload);

$("rep-again").addEventListener("click", () => {
  state.reportFrom === "live" ? enterLive() : enterUpload();
});
$("rep-home").addEventListener("click", () => show("home"));

window.addEventListener("beforeunload", () => leaveLive(false));

boot();
