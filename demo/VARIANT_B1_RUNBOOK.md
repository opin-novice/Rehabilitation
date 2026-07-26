# Variant B1 — Screen-to-Webcam Consistency: Session Runbook

## One-time setup

1. **Confirm Camo works**  
   `python demo/app_camo.py --list` — verify your phone is detected as a virtual camera.

2. **Pin clip timestamps**  
   For each of the 3 YouTube clips, decide which segment to replay.  
   Record exact in/out timestamps here once chosen:

   | Clip | Video | In (s) | Out (s) | Duration (s) |
   |---|---|---|---|---|
   | `kimore_es1_armlift` | "7 Great Shoulder Rehab Exercises" — Dr Jo | 420 | 435 | 15 |
   | `kimore_es3_trunkrot` | "Seated Trunk Rotation" — Dr Jo | 12 | 29 | 17 |
   | `kimore_es5_squat` | "Bodyweight Squat Tutorial" | 29 | 38 | 9 |

3. **Prepare playback**  
   Have the clipped video segments cued and ready to loop on a monitor/TV.  
   Position the phone on a tripod/stand at the straight-on position first.

4. **Synthetic dry-run (verify pipeline)**  
   ```powershell
   python demo/record_take.py --synthetic --clip-id kimore_es1_armlift --viewpoint straight_on
   python demo/record_take.py --synthetic --clip-id kimore_es1_armlift --viewpoint plus30
   python demo/record_take.py --synthetic --clip-id kimore_es1_armlift --viewpoint minus30
   python src/variant_b1_score.py --synthetic-only
   python src/variant_b1_figures.py
   ```
   Confirm end-to-end passes without errors (outputs/variant_b1/ has JSON + figures/).

## Full sweep order (3 clips × 7 viewpoints)

```
clip_id                viewpoint       exercise
──────────────────────────────────────────────
kimore_es1_armlift     straight_on      1
kimore_es1_armlift     plus30           1
kimore_es1_armlift     minus30          1
kimore_es1_armlift     tilt_up          1
kimore_es1_armlift     tilt_down        1
kimore_es1_armlift     closer           1
kimore_es1_armlift     farther          1
──────────────────────────────────────────────
kimore_es3_trunkrot    straight_on      3
kimore_es3_trunkrot    plus30           3
kimore_es3_trunkrot    minus30          3
kimore_es3_trunkrot    tilt_up          3
kimore_es3_trunkrot    tilt_down        3
kimore_es3_trunkrot    closer           3
kimore_es3_trunkrot    farther          3
──────────────────────────────────────────────
kimore_es5_squat       straight_on      5
kimore_es5_squat       plus30           5
kimore_es5_squat       minus30          5
kimore_es5_squat       tilt_up          5
kimore_es5_squat       tilt_down        5
kimore_es5_squat       closer           5
kimore_es5_squat       farther          5
```

## Per-take steps

```powershell
python demo/record_take.py --clip-id kimore_es1_armlift --viewpoint straight_on --duration 30
```

1. Start the video clip on the monitor in a loop.
2. Position phone at the first viewpoint.
3. Run the command above.
4. Wait for recording to finish (or Ctrl+C if done early).
5. **Spot-check** (optional): run the score script on just this one take.

## Live-human control (after full sweep)

Perform the same exercises directly on camera (no screen replay):

```powershell
python demo/record_take.py --clip-id kimore_es1_armlift --viewpoint straight_on --duration 20
python demo/record_take.py --clip-id kimore_es1_armlift --viewpoint plus30 --duration 20
```

Repeat for es3 and es5. These are labelled with the same clip_id but marked `synthetic=false` — the analysis script keeps them as a separate block.

## Post-session scoring + figures

```powershell
python src/variant_b1_score.py
python src/variant_b1_figures.py
```

Check:
- `outputs/variant_b1/consistency_scores.json` — all takes scored
- `outputs/variant_b1/results_table.csv` — cross-viewpoint spread per clip
- `outputs/variant_b1/figures/*.png` — score-vs-time traces

## Expected result

EGRU traces should collapse onto each other across viewpoints (spread ~near-zero, bounded by
pose-estimation noise only). PCT traces should fan out visibly (the model re-sees a "different"
exercise from each angle). InvariantGRU should also show near-zero spread (hand-crafted invariants
are viewpoint-independent by construction).
