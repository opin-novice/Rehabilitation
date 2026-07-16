# Camera Test Guide — Live Webcam Demo

## Overview

This guide walks you through testing the **SE(3)-Equivariant Rehabilitation Assessment** demo
on a machine with a webcam. The demo proves one core thesis: **viewpoint invariance is a theorem,
not a learned tolerance.** When you rotate the camera, our model's score stays flat while the
baseline collapses.

**Time estimate:** 10–15 minutes to set up and run a test; 5 minutes to rehearse the full demo
for the booth.

---

## Part 1: Setup (do this once)

### 1.1 Clone the branch

```bash
git clone -b capstone-showcase https://github.com/opin-novice/Rehabilitation.git
cd Rehabilitation
```

### 1.2 Create a Python environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate
```

### 1.3 Install dependencies

```bash
pip install -r demo/requirements.txt
```

**Optional: NVIDIA GPU support.** If the machine has an NVIDIA GPU and you want it used:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r demo/requirements.txt
```

Otherwise, it runs on CPU (about 1–2 seconds per inference on CPU; the demo is built to be efficient).

### 1.4 Verify without the camera

**Always do this first.** The smoke test runs on a synthetic body and re-measures the thesis, so
you know the models and weights are correct *before* debugging the camera/pose pipeline:

```bash
python demo/smoke_test.py
```

Expected output:

```
EGRU max degradation over rotation : 3.76e-05  (clinical units, /50)
PCT  max degradation over rotation : 8.021

[PASS] EGRU invariance gate: degradation 3.76e-05 < 0.001.
[PASS] Contrast is live: PCT decays 8.021 while EGRU stays flat.
```

If this passes, the science and the checkpoints are good. **Any problems after this are camera
plumbing, not the model.**

---

## Part 2: The 5 KIMORE Exercises

Pick one exercise or do all five in sequence. Each is selectable via keys `1`–`5` in the demo.

### Exercise 1: Trunk Lateral Flexion

**What it is:** Bend your torso sideways (left or right), keeping your hips still.

**How to perform:**
1. Stand upright, feet shoulder-width apart.
2. Slowly bend your torso to the left, reaching your left hand toward your left knee.
3. Hold for 1–2 seconds at the bottom.
4. Return to upright and repeat on the right side.
5. Do 3–5 repetitions slowly.

**Why this exercise:**
- Clear motion in the lateral (side-to-side) plane.
- Very well-tracked by depth sensors.
- Good for first-time testing — the most forgiving.

**Clinical score:** The model rates ROM (range of motion), hip compensation, and symmetry.

---

### Exercise 2: Trunk Forward Flexion

**What it is:** Bend forward, touching your toes (or as far as you can reach).

**How to perform:**
1. Stand upright, feet hip-width apart, legs straight.
2. Slowly bend forward from the hips, lowering your torso toward your legs.
3. Reach as far as comfortable (you don't have to touch your toes).
4. Hold for 1–2 seconds.
5. Return to upright and repeat 3–5 times.

**Why this exercise:**
- Tests the hardest case for depth sensors — the motion is mostly *toward/away* from the camera.
- Even though the scores may be noisier, the invariance property still holds:
  **EGRU stays flat under viewpoint rotation, PCT still decays.**
- This is the real test of the guarantee.

**Clinical score:** Curvature quality (thoracic vs. lumbar bending) and ROM.

---

### Exercise 3: Trunk Rotation

**What it is:** Rotate your torso left and right, keeping your hips and feet facing forward.

**How to perform:**
1. Stand upright, feet shoulder-width apart.
2. Cross your arms on your chest (or hold them out to the sides).
3. Slowly rotate your torso to the left as far as comfortable.
4. Hold for 1–2 seconds.
5. Return to center and rotate right.
6. Do 3–5 repetitions.

**Why this exercise:**
- Pure transverse-plane (axial) rotation.
- Tests scapular stabilization and counter-rotation with the pelvis.
- **Excellent for demonstrating Act 2:** as you press `]` to rotate the *camera* viewpoint, the
  *body's* rotation remains constant, and EGRU stays flat.

**Clinical score:** ROM, pelvic stabilization, and scapular control.

---

### Exercise 4: Hip Abduction

**What it is:** Stand on one leg and lift the other leg out to the side.

**How to perform:**
1. Stand upright on your left leg; use a wall or chair for balance if needed.
2. Lift your right leg straight out to the side, keeping it extended.
3. Hold at the top for 1–2 seconds.
4. Lower your leg back down and repeat 3–5 times.
5. Repeat on the other side (standing on the right leg, lifting the left).

**Why this exercise:**
- Very clear, unambiguous motion in the frontal plane.
- Best-tracked by depth sensors; excellent for learning the system.
- Large, obvious movements that MediaPipe captures reliably.

**Clinical score:** Hip ROM, trunk lean compensation, pelvic stability.

---

### Exercise 5: Hip Circumduction

**What it is:** Stand on one leg and move the lifted leg in slow circles (like drawing a circle in
the air).

**How to perform:**
1. Stand upright on your left leg; hold a wall or chair for balance.
2. Lift your right leg slightly off the ground, keep it straight.
3. Move your leg in slow circles — forward, out to the side, back, and down.
4. Complete 3–5 full circles, then repeat on the other leg.
5. Keep the circles slow and deliberate — jerky motions confuse pose tracking.

**Why this exercise:**
- The most complex motion, testing temporal modeling.
- Combines multiple planes of motion.
- If your model handles this smoothly, the system is working well.

**Clinical score:** Multi-plane ROM, coordination, and stability.

---

## Part 3: Running the Live Demo

### 3.1 Start the app

```bash
python demo/app.py             # default camera 0
python demo/app.py --cam 1     # if you have an external camera plugged in
```

### 3.2 Positioning

1. **Step back from the camera** — make sure your **entire body is visible**, from feet to head.
   MediaPipe needs to see your hips, spine, and shoulders to generate reliable landmarks.
2. **Face the camera** for the first test; once you're comfortable, you can test at angles
   (that's Act 2: the camera rotation challenge).
3. **Good lighting** helps, but the demo works indoors.

### 3.3 Buffer and scoring

1. The app opens a live video window. You should see yourself on screen.
2. **It takes ~2 seconds** to fill the buffer (capturing 64 frames by default).
3. Once the buffer is full, you'll see two score bars appear:
   - **EGRU (ours)** — green bar
   - **PCT (target)** — orange bar
4. The scores update every 6 frames (~6 Hz in real time).

### 3.4 Keyboard controls

| Key | Action | Notes |
|---|---|---|
| `1`–`5` | Select exercise (1=lateral flexion, 2=forward flexion, etc.) | Scores update for the new exercise. |
| `]` | Rotate viewpoint +15° | The skeleton is mathematically rotated; both models score it. EGRU stays flat. |
| `[` | Rotate viewpoint −15° | Same as above, opposite direction. |
| `r` | Auto-sweep 0° → 90° → 0° | Hands-free Act 2. Watch the bars live. |
| `0` | Reset rotation to 0° | Clears any manual rotation. |
| `d` | Toggle frame drops | Simulates a broken camera (Act 3). Rare; use only if curious. |
| `SPACE` | Freeze/unfreeze buffer | Hold a pose for the judges. Useful for booth timing. |
| `q` / `ESC` | Quit | Closes the window. |

---

## Part 4: What to Check

### Check 1: Skeleton Overlay

**What to look for:**
- In the top-right corner, you should see a small skeleton diagram.
- It should track your body as you move — upright, with arms and legs in the right places.
- The skeleton should **stand upright** (vertical spine, not tilted or upside-down).

**If the skeleton is upside-down:**
1. Open `demo/mp_to_kinect.py`.
2. Find the line `AXIS_SIGN = np.array([1.0, -1.0, 1.0])`.
3. Change it to `AXIS_SIGN = np.array([1.0, 1.0, 1.0])` (flip the sign of `AXIS_SIGN[1]`).
4. Run `python demo/app.py` again.

---

### Check 2: EGRU Invariance (The Headline)

**What to look for:**
This is the core thesis. Perform an exercise, then press `]` a few times to rotate the viewpoint.

**Expected:**
- **EGRU bar** (green) does **NOT move** — it stays pinned at the same score.
- **PCT bar** (orange) **visibly swings** — the score changes as the viewpoint rotates.
- The `drift` readout next to each bar shows deviation from the score at 0°:
  - EGRU: `drift ≈ +0.0` (nearly perfect)
  - PCT: `drift ≈ +3 to +8` (noticeably shifts)

**Example (from the smoke test):**
```
angle:  0°   45°   90°  180°
EGRU:  11.6  11.6  11.6  11.6  ← flat
PCT:   40.3  32.7  42.9  32.5  ← swings ±7
```

**If EGRU drifts on your machine:**
- This is **not** a model problem (smoke test proves the model is flat).
- It's a **buffer wiring bug** — likely in how the rotation is applied or how the skeleton is fed
  to the model.
- Check the `engine.py` predict flow and the `rotate_sample` function in `block2_transforms.py`.

---

### Check 3: Latency

**What to look for:**
- The video should update smoothly (~30 Hz from the camera).
- Scores update every 6 frames (~6 Hz).
- If the video is **choppy or stutters**, increase the inference interval.

**If latency is an issue:**
```bash
python demo/app.py --infer-every 10    # run models every 10th frame instead of 6th
python demo/app.py --infer-every 12    # or even fewer, if needed
```

Higher numbers = lower latency, but scores update less frequently (and may lag behind your
motion).

---

## Part 5: Stage Rehearsal (The Booth Demo)

Once all three checks pass, rehearse the full demo for the booth.

### The Script (30–45 seconds)

**Act 1 (10 seconds): Baseline works**
1. Stand upright and perform **Exercise 1 (Lateral Flexion)**.
2. Say: *"This model scores rehabilitation exercises by watching your body in real time. Watch
   the score as I bend sideways."*
3. Narrate the score as it updates.

**Act 2 (20 seconds): Rotating Camera Challenge (THE HEADLINE)**
1. Press `r` to start an auto-sweep of the camera viewpoint (0° → 90° → 0°).
2. **Keep performing the same exercise — don't move, just the camera rotates.**
3. Say: *"Now here's the breakthrough: I'm doing the exact same movement. The camera is rotating
   around me. But watch our model's score — it never changes. The baseline collapses.*
   **Viewpoint invariance is guaranteed by the math, not by the data.**"*
4. Point to the bars: *"Green bar = our model (flat). Orange bar = the baseline (swings). That
   guarantee is why this approach wins."*

**Act 3 (5–10 seconds): Robustness (optional flourish)**
1. If you want to show frame-drop robustness, press `d` to toggle drops.
2. Say: *"Even if the camera glitches and drops frames, the model stays stable."*
3. This is a minor point — the real demo is Act 2.

### Booth Tips

- **Rehearse the script 2–3 times** before the competition. Timing matters.
- **Freeze the buffer** (SPACE) at key moments so judges can see a stable score.
- **Point at the bars** and narrate the contrast. Let the judges see EGRU flat, PCT moving.
- **Have a backup video** of this demo running smoothly (in case of camera/lighting issues at the
  booth). Save it with your phone and keep it in your back pocket.
- **Know the exercise** — if you stumble during Act 1, the whole thing falls flat. Practice
  Exercise 1 until it's muscle memory.

---

## Part 6: Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `cannot open camera 0` | Camera is busy (Zoom, Teams, etc.) or plugged into a different port. | Close other apps; try `--cam 1` or `--cam 2`. |
| Scores stay at `0.0` | Your body is not fully visible to MediaPipe. | Step back so hips and shoulders are in frame. MediaPipe needs to see the full upper body. |
| Video is choppy/stutters | Models running too frequently. | Raise `--infer-every` (e.g., `--infer-every 10`). |
| Skeleton overlay is upside-down | Y-axis is flipped. | Flip `AXIS_SIGN[1]` in `demo/mp_to_kinect.py` (change from `−1.0` to `1.0`). |
| Skeleton doesn't track smoothly | Lighting, occlusion, or MediaPipe confidence too high. | Improve lighting; make sure you're fully visible; try a different angle. |
| Smoke test fails | Real problem with checkpoints or setup. | Stop. Run `python demo/smoke_test.py` again and post the full output. Don't debug the camera yet. |
| EGRU drifts on rotation (Check 2 fails) | Buffer wiring bug (not model bug). | Verify `engine.py` `predict()` and `rotate_sample()` in `block2_transforms.py`. The smoke test proves the model is correct. |
| `model not found: pose_landmarker_full.task` | File not downloaded during clone. | The error message prints a `curl` download command; run it. |

---

## Part 7: Checklist Before the Booth

- [ ] Cloned the `capstone-showcase` branch
- [ ] Ran `python demo/smoke_test.py` → **PASS**
- [ ] Ran `python demo/app.py` with a camera present
- [ ] **Check 1 (skeleton overlay)** — upright, tracks your body
- [ ] **Check 2 (EGRU invariance)** — green bar flat, orange bar swings on rotation
- [ ] **Check 3 (latency)** — video is smooth, no stutter
- [ ] Rehearsed the script 2–3 times
- [ ] Saved a backup video of the full demo (phone recording)
- [ ] Know Exercise 1 (Lateral Flexion) by heart

---

## Part 8: What This Demo Proves

**The claim:** SE(3)-equivariance makes the model viewpoint-invariant by construction (a theorem,
not learned).

**The evidence:** Act 2. You perform the same exercise, the camera rotates, EGRU's score doesn't
move (ΔMAD ≈ 0.00004), while the baseline loses ±8 points. No other architecture can make that
guarantee.

**Why it matters:** Traditional models learn viewpoint tolerance from data — noisy, limited to
their training distribution, and fragile to new camera angles. We've proven the angle out of
the model's latent space entirely. Rotate the camera 180°, retrain nothing, the model still
works.

---

## Questions?

If something doesn't work, check the main demo README:

```
demo/README.md
```

It has additional technical details. But this guide should cover everything you need for a
successful booth demo. Good luck! 🎯
