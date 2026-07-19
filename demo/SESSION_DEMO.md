# Session Demo — "Perform an exercise, get a report"

The Innovation Challenge demo for non-technical judges. **No model comparison.** A person picks an
exercise, performs it for a short timed session, and receives a clean **report card**: an overall
Movement Quality score, a star rating, a breakdown of their movement, and plain-language
**strengths** and **things to improve**.

> This is a separate, additive app. The comparison demos (`app.py`, `app_camo.py`) are untouched.

---

## What the judge sees

1. **Pick an exercise** (keys `1`–`5`) — its **name** and a one-line "how to" appear on screen.
2. **Press SPACE** → a **3-2-1 countdown**, then a **~20-second session**. A live Movement Quality
   bar rises as they move; a timer counts down.
3. **Session ends** → a full-screen **report card** appears (and is saved as a PNG):
   - **MOVEMENT QUALITY /100** + star rating + band (Excellent / Good / Fair / Needs work)
   - a small **AI model score /50** badge (the SE(3)-equivariant model)
   - five explainable metrics: **Range of motion, Smoothness, Symmetry/balance, Tempo & rhythm,
     Consistency** — each a %, a bar, and a verdict
   - **STRENGTHS** (green) and **WHAT TO IMPROVE** (amber), written in plain English

---

## Run it

```bash
python demo/smoke_session.py         # no camera: proves the whole report pipeline + renders a PNG
python demo/app_session.py           # auto-detect camera (webcam OR phone via Camo)
python demo/app_session.py --cam 1   # force a camera index
python demo/app_session.py --list    # list available cameras
python demo/app_session.py --seconds 25   # longer session
```

### Controls
| key | action |
|---|---|
| `1`–`5` | choose the exercise (in the select / between-sessions screen) |
| `SPACE` | start the session · also dismiss the report to start another |
| `e` | end the current session early |
| `q` / `ESC` | quit |

Reports are saved to `demo/reports/report_TIMESTAMP.png` — keep these as booth takeaways.

---

## Making the AI-model-score badge read well (calibration)

The SE(3)-equivariant model was trained on **Kinect depth**. On a **webcam** the input shifts, so
its raw score reads low (~11–15/50). The **headline Movement Quality score does not use the model**
— it's computed from real biomechanics, so it's always sensible. Only the small **AI badge** shows
the model number, and you can calibrate it to your webcam:

```bash
# perform 3 good reference reps of exercise 1; tell it they should read ~42/50
python demo/calibrate.py --exercise 1 --target 42 --sessions 3

# (optional) add weaker reps to anchor the slope, refit without discarding the good ones
python demo/calibrate.py --exercise 1 --target 22 --sessions 2 --append

python demo/calibrate.py --show      # inspect the fitted numbers
python demo/calibrate.py --reset     # back to identity (raw model output)
```

This writes `demo/calibration.json` (a fixed `display = a*raw + b` per exercise). It's an **honest
domain adjustment fitted from your own performances**, not score inflation — repeat per exercise
(`--exercise 1..5`). Ships as identity, so the demo runs truthfully before you calibrate.

---

## Booth script (≈40 seconds)

1. **Select** (5s): "Pick the exercise — say, Trunk Lateral Flexion. It tells you how to do it."
2. **Session** (20s): press SPACE, count down, perform. "It's watching my movement and scoring the
   quality live — range, smoothness, balance, rhythm."
3. **Report** (15s): "And here's the report — an overall Movement Quality score, my star rating,
   what I did well, and what to work on. Just like a physiotherapist's feedback, instantly."

**Honesty framing (say this if a technical judge asks about the score):** *"The headline score is
computed from measured biomechanics — range of motion, smoothness, symmetry, tempo, consistency —
so every number is explainable. The AI badge is our SE(3)-equivariant model, whose real strength is
that its assessment is provably viewpoint-invariant."*

---

## The exercises

| key | exercise | how to perform |
|---|---|---|
| 1 | Trunk Lateral Flexion | Stand tall, bend your torso slowly to each side. |
| 2 | Trunk Forward Flexion | Stand tall, bend forward from the hips and return. |
| 3 | Trunk Rotation | Keep hips forward, rotate your torso left and right. |
| 4 | Hip Abduction | Stand on one leg, lift the other out to the side. |
| 5 | Hip Circumduction | Stand on one leg, draw slow circles with the other. |

Stand far enough back that your **whole body is in frame** (MediaPipe needs hips and shoulders).

---

## How the numbers are computed (for your own reference)

All from the raw skeleton motion, in `demo/feedback.py`:
- **Range of motion** — peak-to-peak of the exercise's primary joint angle vs a target excursion.
- **Smoothness** — spectral arc length (SPARC) of movement speed (tremor/jerk → lower score).
- **Symmetry/balance** — left-vs-right limb ROM, or the two movement directions, or shoulder levelness.
- **Tempo & rhythm** — repetition count + cadence in a healthy band + spectral rhythm consistency.
- **Consistency** — how alike each repetition's amplitude is (motor control).
- **Movement Quality** — the (equally-weighted) average of the five, 0–100, → star rating.
