# RehabSense — Web Frontend

A browser UI for the rehab assessment pipeline, with two modes:

1. **Live Session** — perform an exercise in front of your webcam; a skeleton tracks you in
   real time, a live Movement Quality score updates as you move, and a full report card
   appears when the timed session ends.
2. **Upload a Video** — drop in a recorded clip; the same pipeline extracts poses from every
   frame (real video timestamps — irregular sampling is native to the model) and produces
   the identical report.

Both modes share one scoring path: MediaPipe pose → Kinect-25 remap ([demo/mp_to_kinect.py](../demo/mp_to_kinect.py))
→ explainable biomechanics ([demo/feedback.py](../demo/feedback.py)) for the headline
Movement Quality composite, plus the calibrated **SE(3)-equivariant EGRU** ensemble
([demo/engine.py](../demo/engine.py)) as the AI model badge. The report is also rendered to a
downloadable PNG via [demo/report_card.py](../demo/report_card.py).

## Run

```bash
pip install -r demo/requirements.txt        # torch, e3nn, mediapipe, opencv, ...
pip install -r frontend/requirements.txt    # fastapi, uvicorn, multipart, pillow

python frontend/server.py                   # open http://127.0.0.1:8000
python frontend/server.py --port 9000       # alternative port
```

Requires the checkpoints in `outputs/cde_block2/` (`egru_s0_pooled_f{0..4}.pt`) and the
MediaPipe model `demo/models/pose_landmarker_full.task` — both are committed on the
`capstone-showcase` branch.

> **Camera note:** browsers only allow webcam access on `localhost` or HTTPS. The default
> `127.0.0.1:8000` works out of the box. To use a phone or another machine on your LAN,
> serve behind HTTPS (e.g. `caddy reverse-proxy`) or use a Chrome
> `unsafely-treat-insecure-origin-as-secure` flag for the LAN origin.

## Architecture

```
frontend/
  server.py          FastAPI backend
  static/
    index.html       single-page app (home → exercise → live/upload → report)
    style.css        dark clinical theme
    app.js           camera pump, WebSocket client, upload/polling, report rendering
```

- `WS /ws/live` — browser streams ~15 fps JPEG frames (ack-paced, so a slow CPU degrades
  gracefully instead of piling up); server runs a per-connection MediaPipe VIDEO-mode
  tracker and replies with Kinect-25 overlay points + a live composite score. `start` /
  `stop` messages bracket the scored session; `stop` returns the full report JSON.
- `POST /api/upload` (+ `GET /api/job/{id}`) — background thread decodes the video at up to
  ~15 fps effective (first 3 minutes), extracts poses with true video timestamps, then runs
  the shared finalize path. Progress is polled.
- `GET /api/exercises`, `/api/health`, `/api/bones` — UI metadata.
- `/reports/*` — the rendered report PNGs (saved under `demo/reports/`).

## Troubleshooting

| symptom | fix |
|---|---|
| "server offline" pill | start `python frontend/server.py`; check the port |
| "AI model failed to load" | `pip install e3nn==0.6.0`; check `outputs/cde_block2/*.pt` exist |
| camera blocked | allow permission; close Zoom/Teams; use `localhost`, not a LAN IP |
| "no person detected" | whole body must be visible — hips **and** shoulders |
| live video stutters | close other tabs; the frame pump auto-throttles, scores still work |
