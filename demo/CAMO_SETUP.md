# Camo Setup Guide — Test the Demo on This PC

Use this guide if you're testing the demo on a Windows/Mac PC with an iPhone/Android phone as
the webcam via the **Camo** app.

## 1. Install Camo

Camo allows you to use your phone as a wireless webcam. It works on Windows, Mac, and Linux.

### 1.1 On your phone (iPhone/Android)

1. Install the **Camo** app from the App Store (iOS) or Google Play (Android).
2. Open the app and sign in (or create an account if needed).

### 1.2 On the PC (Windows/Mac)

1. Download Camo from https://reincubate.com/camo/
2. Install it on your PC.
3. Launch Camo on your PC. It will prompt you to connect your phone.
4. On your phone, open the Camo app. Your PC should appear in the list of available devices.
5. Tap to connect. You should see a live preview on the PC.

## 2. Test that Camo appears as a camera device

Once Camo is connected, it registers as a **virtual camera** that OpenCV can access.

Run this quick Python test:

```python
import cv2

# List all available cameras
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i}: works (frame shape: {frame.shape})")
            cap.release()
        else:
            print(f"Camera {i}: exists but no frames")
            cap.release()
    else:
        # Camera doesn't exist; stop checking
        if i > 0:
            break
```

Example output:
```
Camera 0: works (frame shape: (1080, 1920, 3))
Camera 1: exists but no frames
```

Note the **camera index** where Camo appears (usually 0 or 1). You'll pass this to `app_camo.py`.

## 3. Run the demo with Camo

```bash
# default Camo camera (usually 0)
python demo/app_camo.py

# or specify the camera index if Camo is on a different device
python demo/app_camo.py --cam 1

# other options work the same as app.py
python demo/app_camo.py --infer-every 10  # slower inference for lower latency
python demo/app_camo.py --window 48       # smaller rolling buffer
```

## 4. Camo-specific quirks

### Frame rate
Camo streams at ~30 Hz by default over WiFi. This is fine for the demo — inference happens every
6th frame anyway (~6 Hz).

### Resolution
Camo typically sends 1080p or lower depending on your phone and network quality. The demo
auto-detects and works at any resolution.

### Lag
If there's noticeable lag, it's usually the WiFi connection, not the code:
- Make sure your phone and PC are on the **same WiFi network** (not 2.4 vs 5 GHz splitting).
- Move closer to the router.
- Close other bandwidth-heavy apps (video calls, downloads).

### Orientation
Camo respects your phone's orientation. If the image is sideways or upside-down, rotate your
phone and Camo will adjust.

## 5. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| Camo app won't connect to PC | Network issue or firewall blocking | Make sure both are on the same WiFi; restart Camo on both; check Windows Firewall allows Camo |
| Camera shows no frames (`cap.isOpened()` is False) | Wrong camera index or Camo not running | Run the test script above; make sure Camo is connected and streaming |
| Video is very laggy | Poor WiFi or phone far from router | Move phone closer; check WiFi signal strength; close other apps |
| Image is sideways/upside-down | Phone orientation | Rotate your phone; Camo will auto-adjust |
| Skeleton overlay is garbled or missing | Frame too large/small for MediaPipe | Try a different resolution in Camo settings (1080p usually works best) |

## 6. Development iteration workflow

For testing the demo on THIS PC (without a real webcam):

```bash
# 1. Verify models are correct (no Camo needed)
python demo/smoke_test.py

# 2. Start Camo on your phone and connect to this PC
# (use the Camo PC app)

# 3. Run the demo with Camo
python demo/app_camo.py

# 4. Perform exercises and test the invariance (press ] to rotate)
```

Once this PC's Camo path is verified, you can download the `capstone-showcase` branch on the
webcam PC and run the normal `python demo/app.py`.

---

## 7. Next: download to the webcam PC

Once you've verified the demo works on this PC with Camo, clone the `capstone-showcase` branch
on the machine with a real webcam:

```bash
git clone -b capstone-showcase https://github.com/opin-novice/Rehabilitation.git
cd Rehabilitation
pip install -r demo/requirements.txt
python demo/app.py  # standard webcam (no Camo needed)
```

See `demo/CAMERA_TEST_GUIDE.md` for full instructions on the webcam PC.
