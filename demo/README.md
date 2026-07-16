# Live Webcam Demo — SE(3)-Equivariant Rehab Assessment

**Branch: `capstone-showcase`.** This branch is self-contained: it carries the demo code *and*
the trained model weights, so you can clone it onto a machine with a webcam and run it without
any dataset, training, or extra downloads.

**The headline:** viewpoint invariance is a **theorem**, not a learned tolerance. Rotate the
viewpoint live and our model (EGRU) stays flat while the state-of-the-art target paper (PCT)
collapses. That contrast *is* the demo.

---

## 1. Quickstart on the webcam PC

```bash
git clone -b capstone-showcase https://github.com/opin-novice/Rehabilitation.git
cd Rehabilitation

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r demo/requirements.txt
```

Optional: if that PC has an NVIDIA GPU and you want it used, install CUDA torch *first*
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`), then the line above.
**A GPU is not required** — the demo is built to run on CPU.

### Step 1 — verify without the camera (do this first)

```bash
python demo/smoke_test.py
```

This runs the full pipeline on a synthetic body and re-measures the thesis. Expect:

```
EGRU max degradation over rotation : 3.76e-05   (clinical units, /50)
PCT  max degradation over rotation : 8.021
[PASS] EGRU invariance gate: degradation 3.76e-05 < 0.001.
[PASS] Contrast is live: PCT decays 8.021 while EGRU stays flat.
```

If this passes, the models and weights are correct on that machine. **Any problem after this
point is camera/pose plumbing, not the science.**

### Step 2 — run the live demo

```bash
python demo/app.py             # default camera 0
python demo/app.py --cam 1     # external/second webcam
```

Stand back far enough that your **whole body is in frame** (MediaPipe needs hips and shoulders
to produce the 3D world landmarks the models consume). Give it ~2 seconds to fill the buffer —
scores appear once 24 frames are captured.

---

## 2. Controls (the three acts)

| key | action |
|---|---|
| `1`–`5` | pick which KIMORE exercise is being scored |
| `]` / `[` | rotate viewpoint +/- 15° (**Act 2 — the headline**) |
| `r` | auto-sweep rotation 0→90→0, hands-free |
| `d` | toggle frame drops (**Act 3** robustness) |
| `SPACE` | freeze/unfreeze the buffer (hold a pose for the judges) |
| `0` | reset rotation to 0 |
| `q` / `ESC` | quit |

---

## 3. What to check on the webcam PC

Three things, in order. I could not test these here — this machine has no camera.

1. **The skeleton overlay tracks you and stands upright** (top-right corner box).
   If it renders upside-down, flip the sign of `AXIS_SIGN[1]` in `demo/mp_to_kinect.py`.
   The smoke test already fixes Y-up, so it should be correct.

2. **Press `]` a few times. The EGRU bar must NOT move; the PCT bar must move.**
   This is the whole demo. It is guaranteed by construction, so if the EGRU bar drifts,
   something in the *buffer wiring* is wrong — not the model (the smoke test proves the model
   is flat to 3.8e-05). The `drift` readout next to each bar shows the deviation from the
   score at 0°: EGRU should read ~`+0.0`, PCT should visibly swing.

3. **Latency.** Models run every 6th frame by default. If the video stutters, raise it:
   `python demo/app.py --infer-every 10`.

### Troubleshooting

| symptom | fix |
|---|---|
| `cannot open camera 0` | try `--cam 1`, or close Zoom/Teams/anything holding the camera |
| scores stay at `0.0` | your full body isn't in frame — step back; MediaPipe needs hips + shoulders |
| video is choppy | raise `--infer-every` (e.g. `10`), or shrink `--window` |
| `model not found: ...pose_landmarker_full.task` | the file is committed on this branch; re-clone, or use the `curl` line the error prints |
| smoke test fails | stop — that's a real problem, don't debug the camera. Send me the output. |

---

## 4. What's in this branch

Bundled so the demo needs no setup beyond `pip install`:

- `demo/models/pose_landmarker_full.task` (9 MB) — MediaPipe pose model.
- `outputs/cde_block2/egru_s0_pooled_f{0..4}.pt` (2 MB each) — our 5-fold EGRU ensemble.
- `outputs/cde_block2/pct_pooled_f{0..4}.pt` (21 MB each) — the 5-fold PCT baseline ensemble.

These are normally gitignored (`outputs/`) and are force-added **on this branch only**, so the
showcase PC gets a working clone. Don't merge this branch into `main`.

### Architecture

- `mp_to_kinect.py` — MediaPipe 33 landmarks → Kinect-25 skeleton + training-matched preprocess.
- `pose_backend.py` — MediaPipe Tasks `PoseLandmarker` wrapper (the legacy `mp.solutions` API is
  gone in mediapipe ≥ 0.10.3x).
- `engine.py` — loads both 5-fold ensembles; `predict(sample, exercise, angle)` feeds
  **byte-identical** input to both models, so the side-by-side is honest.
- `app.py` — OpenCV live loop (the robust booth fallback).
- `smoke_test.py` — the invariance gate, re-measured on every run.

---

## 5. Honesty note for the pitch

We headline the **structural guarantee** (invariance / robustness), not a clinical-accuracy
claim. Act 2's win does **not** depend on the live webcam scores being clinical-grade — EGRU's
invariance is architectural, so it holds under any input, including a domain shift from Kinect
to webcam. Say "the score cannot change when the camera moves, by construction" — that is the
defensible claim and no baseline can match it.
