#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
server.py
=========
FastAPI backend for the web frontend. Wraps the SAME pipeline the OpenCV demos use
(demo/mp_to_kinect.py, demo/feedback.py, demo/engine.py, demo/report_card.py) behind
two entry points:

  1. LIVE SESSION  --  WS /ws/live
     The browser streams JPEG frames; the server runs MediaPipe pose (VIDEO mode,
     one stateful tracker per connection), buffers the Kinect-25 skeleton while a
     session is recording, pushes back 2-D overlay points + a live Movement Quality
     score, and on "stop" returns the full report (biomechanics composite + the
     calibrated SE(3)-equivariant EGRU badge) plus a rendered PNG.

  2. VIDEO UPLOAD  --  POST /api/upload  ->  GET /api/job/{id}
     The uploaded file is decoded with OpenCV at up to ~15 fps effective, poses are
     extracted with real video timestamps (the model is built for irregular
     sampling, so dropped/skipped frames are fine), and the identical report is
     produced. Progress is polled.

Run:  python frontend/server.py            (then open http://127.0.0.1:8000)
      python frontend/server.py --port 9000
"""

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
import sys
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "demo"), os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import mediapipe as mp                                                # noqa: E402
from mediapipe.tasks import python as mp_python                       # noqa: E402
from mediapipe.tasks.python import vision as mp_vision                # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket  # noqa: E402
from fastapi.staticfiles import StaticFiles                           # noqa: E402
from starlette.websockets import WebSocketDisconnect                  # noqa: E402

import feedback as fb                                                 # noqa: E402
import report_card as rc                                              # noqa: E402
from mp_to_kinect import MP, mediapipe_to_kinect25, preprocess_window  # noqa: E402
from pose_backend import MODEL_PATH as POSE_MODEL_PATH                # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
REPORTS_DIR = os.path.join(ROOT, "demo", "reports")
CKPT_DIR = os.path.join(ROOT, "outputs", "cde_block2")

MIN_FRAMES = 8                # fewer than this -> "no movement captured"
LIVE_EVERY = 6                # recompute the live composite every N buffered frames
LIVE_MIN = 20                 # frames needed before the first live score
MAX_SESSION_S = 120.0         # server-side cap on a live session buffer
MAX_VIDEO_S = 180.0           # only the first 3 minutes of an upload are analyzed
TARGET_FPS = 15.0             # pose-extraction rate for uploads
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}

# Same bone subset the OpenCV demos draw (Kinect-25 indices).
DRAW_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7),
    (20, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

@asynccontextmanager
async def _lifespan(_app):
    """Startup work. Replaces the deprecated @app.on_event("startup") hook.

    The engine load stays on a daemon thread rather than blocking here: the UI is
    meant to come up immediately and report `engine: loading` on /api/health while
    the five EGRU folds arrive.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    threading.Thread(target=_load_engine, daemon=True).start()
    yield


app = FastAPI(title="RehabSense", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# model engine (loaded once, in the background, at startup)
# ---------------------------------------------------------------------------
ENGINE = None
ENGINE_ERR = None
ENGINE_LOCK = threading.Lock()      # EGRU inference is not re-entrant across threads


def _load_engine():
    global ENGINE, ENGINE_ERR
    try:
        from engine import DemoEngine
        eng = DemoEngine(ckpt_dir=CKPT_DIR, load_pct=False)
        ENGINE = eng
        print(f"[server] EGRU ensemble ready on {eng.device}")
    except Exception as e:                                            # pragma: no cover
        ENGINE_ERR = f"{type(e).__name__}: {e}"
        print(f"[server] ENGINE FAILED TO LOAD: {ENGINE_ERR}")


def _predict_ai(sample, exercise):
    """EGRU ensemble score in clinical units (/50), or None if the engine is down."""
    if ENGINE is None:
        return None
    s = dict(sample)
    s["exercise"] = int(exercise)
    s.setdefault("y", 0.0)
    with ENGINE_LOCK:
        return ENGINE.predict_egru(s)


# --- validation thresholds (calibrated on real vs. random video; see scratch diagnose) ---
#   real lateral-flexion clip : coverage 100%, max plane excursion 48 deg
#   random cats/cartoons clip : coverage 47%,  max plane excursion 271 deg (impossible)
MIN_COVERAGE = 0.60          # tracked / attempted frames; below this the content isn't one person
MAX_EXCURSION_DEG = 130.0    # a real trunk/limb swing can't exceed this; garbage skeletons do
MIN_PLANE_DEG = 8.0          # the claimed exercise's own plane must show at least this much motion
PLANE_DOMINANCE = 0.45       # claimed plane must be >= this fraction of the biggest plane

# which anatomical plane each exercise primarily lives in
EXERCISE_CHANNEL = {1: "frontal", 2: "sagittal", 3: "transverse", 4: "limb", 5: "limb"}
CHANNEL_EXERCISE = {"frontal": "Trunk Lateral Flexion", "sagittal": "Trunk Forward Flexion",
                    "transverse": "Trunk Rotation", "limb": "a hip exercise"}


def _plane_signature(x):
    """Movement excursion (deg, robust p97-p3) in each anatomical channel.
    Same joint conventions as feedback.py; used only to sanity-check the CLAIMED exercise."""
    def exc(s):
        return float(np.percentile(s, 97) - np.percentile(s, 3)) if len(s) >= 3 else 0.0

    spine = x[:, 20] - x[:, 0]                                    # SpineShoulder - SpineBase
    frontal = np.degrees(np.arctan2(spine[:, 0], spine[:, 1]))    # side bend
    sagittal = np.degrees(np.arctan2(spine[:, 2], spine[:, 1]))   # forward bend
    sho, hip = x[:, 8] - x[:, 4], x[:, 16] - x[:, 12]             # R-L shoulder / hip
    rot = np.degrees(np.arctan2(sho[:, 2], sho[:, 0])) - np.degrees(np.arctan2(hip[:, 2], hip[:, 0]))
    rot = (rot + 180) % 360 - 180

    def leg(knee, hipj):
        v = x[:, knee] - x[:, hipj]
        n = np.linalg.norm(v, axis=-1)
        cs = np.clip((v @ np.array([0.0, -1.0, 0.0])) / np.clip(n, 1e-8, None), -1, 1)
        return np.degrees(np.arccos(cs))
    limbL, limbR = leg(13, 12), leg(17, 16)
    limb = limbL if exc(limbL) >= exc(limbR) else limbR
    return {"frontal": exc(frontal), "sagittal": exc(sagittal),
            "transverse": exc(rot), "limb": exc(limb)}


def _validate(x, rep, exercise, n_tracked, n_attempts):
    """Reject inputs that aren't a single person deliberately doing THIS exercise.
    Raises ValueError (user-facing message) on failure. Grounded in measured thresholds,
    NOT arbitrary guesses. Note: lateral-flexion vs. rotation genuinely overlap and are
    intentionally NOT separated here (that needs a trained classifier)."""
    # 1. human presence: enough frames actually contained a trackable body
    cov = n_tracked / max(n_attempts, 1)
    if cov < MIN_COVERAGE:
        raise ValueError(
            f"only {cov*100:.0f}% of the video contained a trackable person "
            f"(need {MIN_COVERAGE*100:.0f}%+). Make sure one person's whole body is "
            "visible for the whole clip — not cartoons, animals, or mixed footage.")

    sig = _plane_signature(x)
    dominant_ch = max(sig, key=sig.get)
    dominant = sig[dominant_ch]

    # 2. anatomical plausibility: real trunk/limb swings can't exceed ~130 deg
    if dominant > MAX_EXCURSION_DEG:
        raise ValueError(
            f"the detected motion is not anatomically plausible (a {dominant:.0f}° swing). "
            "The skeleton is jumping between different bodies/scenes — this isn't one "
            "person performing an exercise.")

    # 3. movement must match the CLAIMED exercise's plane (clear mismatches only)
    ch = EXERCISE_CHANNEL[exercise]
    claimed = sig[ch]
    if claimed < MIN_PLANE_DEG:
        raise ValueError(
            f"there's almost no movement in the plane this exercise uses "
            f"({claimed:.0f}° in the {ch} plane). This doesn't look like "
            f"{fb.EXERCISE_NAMES[exercise]}.")
    if claimed < PLANE_DOMINANCE * dominant and claimed < dominant:
        raise ValueError(
            f"the movement doesn't match {fb.EXERCISE_NAMES[exercise]} — most of the "
            f"motion is in the {dominant_ch} plane ({dominant:.0f}°), which looks more "
            f"like {CHANNEL_EXERCISE[dominant_ch]}, not the {ch} plane "
            f"({claimed:.0f}°) this exercise needs.")

    # 4. repetition: a real session has a repeated pattern
    consistency = {m['key']: m['pct'] for m in rep['metrics']}.get('consistency', 50)
    if consistency < 12 and rep['reps'] < 2:
        raise ValueError(
            "no repeated movement pattern was found. Perform several clear, "
            "deliberate repetitions of the exercise.")
    return True


def _finalize(sess_x, sess_t, exercise, n_tracked=None, n_attempts=None):
    """Buffers -> (report dict, png url). The one scoring path both modes share."""
    x = np.stack(sess_x)
    t = np.asarray(sess_t, dtype=np.float64)
    n_tracked = n_tracked if n_tracked is not None else len(sess_x)
    n_attempts = n_attempts if n_attempts is not None else len(sess_x)

    sample = preprocess_window(x, t)
    ai_raw = _predict_ai(sample, exercise)
    rep = fb.compute_report(x, t, int(exercise), ai_score_raw=ai_raw)

    _validate(x, rep, int(exercise), n_tracked, n_attempts)      # raises ValueError on reject

    try:
        _, png_path = rc.render_report(rep, save=True)
        png_url = "/reports/" + os.path.basename(png_path) if png_path else None
    except Exception:
        png_url = None
    rep["png"] = png_url
    return rep


# ---------------------------------------------------------------------------
# pose tracking (per-session MediaPipe VIDEO-mode tracker, world + image points)
# ---------------------------------------------------------------------------
class PoseSession:
    """Like demo.pose_backend.PoseBackend, but also returns the normalized IMAGE
    landmarks so the browser can draw the skeleton over its own video element."""

    def __init__(self):
        if not os.path.exists(POSE_MODEL_PATH):
            raise RuntimeError(f"pose model missing: {POSE_MODEL_PATH}")
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False)
        self._lm = mp_vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, rgb_uint8, timestamp_ms):
        """-> (kinect25_world (25,3) metres, kinect25_image (25,2) normalized) or (None, None)."""
        ts = int(timestamp_ms)
        if ts <= self._last_ts:                       # VIDEO mode demands monotonic stamps
            ts = self._last_ts + 1
        self._last_ts = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(rgb_uint8))
        res = self._lm.detect_for_video(img, ts)
        if not res.pose_world_landmarks:
            return None, None
        world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]],
                         dtype=np.float64)
        image = np.array([[p.x, p.y] for p in res.pose_landmarks[0]], dtype=np.float64)
        return mediapipe_to_kinect25(world), _kinect25_image_pts(image)

    def close(self):
        self._lm.close()


def _kinect25_image_pts(lm33):
    """MediaPipe 33 normalized image landmarks -> Kinect-25 2-D points for the overlay.
    Same joint synthesis as mediapipe_to_kinect25, minus the world-axis flip (these are
    already screen coordinates)."""
    w = lm33
    j = np.zeros((25, 2), dtype=np.float64)
    hip_mid = 0.5 * (w[MP["l_hip"]] + w[MP["r_hip"]])
    sho_mid = 0.5 * (w[MP["l_sho"]] + w[MP["r_sho"]])
    head = 0.5 * (w[MP["l_ear"]] + w[MP["r_ear"]])
    j[0], j[20], j[1], j[2], j[3] = hip_mid, sho_mid, 0.5 * (hip_mid + sho_mid), \
        0.5 * (sho_mid + head), head
    j[4], j[5], j[6] = w[MP["l_sho"]], w[MP["l_elb"]], w[MP["l_wri"]]
    j[7] = 0.5 * (w[MP["l_pinky"]] + w[MP["l_index"]])
    j[21], j[22] = w[MP["l_index"]], w[MP["l_thumb"]]
    j[8], j[9], j[10] = w[MP["r_sho"]], w[MP["r_elb"]], w[MP["r_wri"]]
    j[11] = 0.5 * (w[MP["r_pinky"]] + w[MP["r_index"]])
    j[23], j[24] = w[MP["r_index"]], w[MP["r_thumb"]]
    j[12], j[13], j[14], j[15] = w[MP["l_hip"]], w[MP["l_knee"]], w[MP["l_ank"]], w[MP["l_foot"]]
    j[16], j[17], j[18], j[19] = w[MP["r_hip"]], w[MP["r_knee"]], w[MP["r_ank"]], w[MP["r_foot"]]
    return j


# ---------------------------------------------------------------------------
# small JSON endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"engine": "ready" if ENGINE is not None else ("error" if ENGINE_ERR else "loading"),
            "engine_error": ENGINE_ERR}


@app.get("/api/exercises")
def exercises():
    return [{"id": i, "name": fb.EXERCISE_NAMES[i], "howto": fb.HOWTO[i]}
            for i in sorted(fb.EXERCISE_NAMES)]


@app.get("/api/bones")
def bones():
    return DRAW_BONES


# ---------------------------------------------------------------------------
# live session over WebSocket
# ---------------------------------------------------------------------------
# Live recording buffers live HERE, keyed by a session id, NOT in the WebSocket
# handler's scope -- so if the socket drops mid-session the frames survive and the
# client can reconnect + resume instead of losing the whole session.
LIVE_SESSIONS = {}                 # sid -> {sid,x,t,miss,exercise,t0_wall,last}
LIVE_SESSIONS_LOCK = threading.Lock()
LIVE_TTL = 90.0                    # a disconnected recording is resumable for this long


def _live_gc():
    now = time.monotonic()
    with LIVE_SESSIONS_LOCK:
        for k in [k for k, v in LIVE_SESSIONS.items() if now - v["last"] > LIVE_TTL]:
            LIVE_SESSIONS.pop(k, None)


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    pose = await asyncio.to_thread(PoseSession)
    conn_t0 = time.monotonic()        # per-connection clock for the MediaPipe detector
    sid = None                        # id of the active session in LIVE_SESSIONS (or None)
    live_busy = False

    async def send(obj):
        await ws.send_text(json.dumps(obj))

    async def live_update(s):
        """Recompute the biomechanics composite on a snapshot of the session buffer."""
        nonlocal live_busy
        try:
            x = np.stack(s["x"])
            t = np.asarray(s["t"], dtype=np.float64)
            rep = await asyncio.to_thread(fb.compute_report, x, t, s["exercise"])
            if sid == s["sid"]:
                await send({"type": "live", "score": rep["composite"],
                            "reps": rep["reps"]})
        except Exception:
            pass
        finally:
            live_busy = False

    async def do_finish(s):
        """Detach the session from the store and score it (runs once per session)."""
        nonlocal sid
        sid = None
        with LIVE_SESSIONS_LOCK:
            LIVE_SESSIONS.pop(s["sid"], None)
        await _finish(ws, send, s["x"], s["t"], s["exercise"], len(s["x"]) + s["miss"])

    await send({"type": "ready"})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                buf = np.frombuffer(msg["bytes"], dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    await send({"type": "pose", "detected": False})
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_ts = (time.monotonic() - conn_t0) * 1000.0
                kin, pts2d = await asyncio.to_thread(pose.detect, rgb, mp_ts)
                s = LIVE_SESSIONS.get(sid) if sid else None
                if kin is None:
                    if s is not None:
                        s["miss"] += 1
                        s["last"] = time.monotonic()
                    await send({"type": "pose", "detected": False})
                    continue
                if s is not None:
                    # session-relative time (monotonic, survives reconnects)
                    s["t"].append(time.monotonic() - s["t0_wall"])
                    s["x"].append(kin)
                    s["last"] = time.monotonic()
                    n = len(s["x"])
                    if n >= LIVE_MIN and n % LIVE_EVERY == 0 and not live_busy:
                        live_busy = True
                        asyncio.get_running_loop().create_task(live_update(s))
                    if s["t"][-1] - s["t"][0] > MAX_SESSION_S:
                        await do_finish(s)
                await send({"type": "pose", "detected": True,
                            "pts": np.round(pts2d, 4).tolist()})

            elif msg.get("text") is not None:
                try:
                    cmd = json.loads(msg["text"])
                except ValueError:
                    continue
                typ = cmd.get("type")
                if typ == "ping":
                    await send({"type": "pong"})
                elif typ == "start":
                    _live_gc()
                    ex = int(cmd.get("exercise", 1))
                    sid = uuid.uuid4().hex[:12]
                    with LIVE_SESSIONS_LOCK:
                        LIVE_SESSIONS[sid] = {"sid": sid, "x": [], "t": [], "miss": 0,
                                              "exercise": ex, "t0_wall": time.monotonic(),
                                              "last": time.monotonic()}
                    await send({"type": "recording", "sid": sid})
                elif typ == "resume":
                    _live_gc()
                    want = cmd.get("sid")
                    if want and want in LIVE_SESSIONS:
                        sid = want
                        LIVE_SESSIONS[sid]["last"] = time.monotonic()
                        await send({"type": "resumed", "sid": sid,
                                    "exercise": LIVE_SESSIONS[sid]["exercise"]})
                    else:
                        sid = None
                        await send({"type": "resume_failed"})
                elif typ == "stop":
                    s = LIVE_SESSIONS.get(sid) if sid else None
                    if s is not None:
                        await do_finish(s)
                    else:
                        await send({"type": "error",
                                    "msg": "Your session couldn't be found — the "
                                           "connection was lost too long to recover. "
                                           "Please start a new session."})
                elif typ == "abort":
                    if sid:
                        with LIVE_SESSIONS_LOCK:
                            LIVE_SESSIONS.pop(sid, None)
                        sid = None
    except WebSocketDisconnect:
        pass
    finally:
        # NOTE: do NOT delete the session here -- leave it in LIVE_SESSIONS so a
        # reconnect within LIVE_TTL can resume it. _live_gc reaps it if abandoned.
        await asyncio.to_thread(pose.close)


async def _finish(ws, send, sess_x, sess_t, exercise, n_attempts=None):
    if len(sess_x) < MIN_FRAMES:
        await send({"type": "error",
                    "msg": "Not enough movement captured — make sure your whole "
                           "body is in frame, then try again."})
        return
    await send({"type": "analyzing"})
    try:
        rep = await asyncio.to_thread(_finalize, sess_x, sess_t, exercise,
                                      len(sess_x), n_attempts or len(sess_x))
        await send({"type": "report", "report": rep})
    except ValueError as e:
        await send({"type": "error", "msg": f"{e}"})
    except Exception as e:
        await send({"type": "error", "msg": f"Scoring failed: {type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# video upload -> background job -> polled report
# ---------------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()


def _job_set(job_id, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw)


def _process_video(job_id, path, exercise):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("could not decode this video file")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1 else 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, int(round(fps / TARGET_FPS)))
        max_frames = int(MAX_VIDEO_S * fps)

        pose = PoseSession()
        sess_x, sess_t = [], []
        idx, attempts = 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or idx >= max_frames:
                    break
                if idx % step == 0:
                    attempts += 1                    # a frame we tried to detect a pose in
                    t_s = idx / fps
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    kin, _ = pose.detect(rgb, t_s * 1000.0)
                    if kin is not None:
                        sess_t.append(t_s)
                        sess_x.append(kin)
                    if total:
                        _job_set(job_id, progress=0.9 * min(idx / max(total, 1), 1.0),
                                 stage=f"extracting poses  ({len(sess_x)} frames tracked)")
                idx += 1
        finally:
            cap.release()
            pose.close()

        if len(sess_x) < MIN_FRAMES:
            raise RuntimeError("no person detected in this video — the whole body "
                               "must be visible")
        _job_set(job_id, progress=0.92, stage="scoring with the AI model")
        rep = _finalize(sess_x, sess_t, exercise, n_tracked=len(sess_x), n_attempts=attempts)
        _job_set(job_id, state="done", progress=1.0, stage="done", report=rep)
    except ValueError as e:                          # validation rejection (user-fixable)
        _job_set(job_id, state="error", error=f"Video didn't pass checks: {e}")
    except Exception as e:
        _job_set(job_id, state="error", error=f"{e}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), exercise: int = Form(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"unsupported file type '{ext}' — use one of "
                                 f"{sorted(ALLOWED_EXT)}")
    if not (1 <= exercise <= 5):
        raise HTTPException(422, "exercise must be 1..5")
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"state": "processing", "progress": 0.0,
                        "stage": "opening video", "report": None, "error": None}
    threading.Thread(target=_process_video, args=(job_id, path, exercise),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return dict(job)


# static mounts LAST so /api and /ws keep precedence
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser(description="RehabSense web frontend server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"[server] open  http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
