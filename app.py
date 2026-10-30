"""
Pumpkin Cam - live pumpkin detector

How it decides:
  1. YOLO-World finds round/orange objects ("candidates") in each frame.
  2. CLIP looks at each candidate up close and decides: pumpkin, orange, ball...
  3. A pumpkin must show up a few times before the banner turns green,
     and the banner stays green for a couple of seconds so it doesn't flicker.

Run:   conda activate pumpkin
       python app.py
Quit:  press Q in the video window
"""
import os
import sys
import time
import math
import threading
import subprocess
from collections import deque

import cv2
from ultralytics import YOLOWorld

# =====================  SETTINGS (tweak these)  =====================
CAMERA_INDEX = 0          # 0 = laptop camera. Try 1 if it opens the wrong one (e.g. iPhone)
IMG_SIZE = 320            # 320 = faster (good for Intel Mac), 480 = finds smaller/farther objects
DETECT_CONF = 0.10        # how sure YOLO must be to even consider an object (low = finds more)
USE_CLIP = True           # CLIP decides pumpkin vs. orange. Keep this True
CLIP_MIN_SCORE = 0.50     # how sure CLIP must be that it's a pumpkin (0 to 1)
CLIP_EVERY = 3            # re-check objects with CLIP every N frames (higher = smoother video)
MAX_OBJECTS = 3           # check at most this many objects per frame (speed)

FRAMES_TO_REMEMBER = 6    # look at the last 6 frames...
FRAMES_NEEDED = 3         # ...and need a pumpkin in 3 of them before turning green
HOLD_SECONDS = 2.0        # once green, stay green this long after the pumpkin was last seen
ALERT_COOLDOWN = 10       # seconds between alerts (beep + photo + Discord)

USE_GEMMA = True          # Gemma 4 gives a final verdict + confidence + reason on each alert
GEMMA_MODEL = "gemma4:e2b"   # smallest Gemma 4 with vision. "gemma4:e4b" = smarter but slower
GEMMA_VETO = False        # True = wait for Gemma and skip the alert if it says "not a pumpkin"
                          # False = snapshot card sends instantly, Gemma's verdict follows after
GEMMA_TIMEOUT = 120       # seconds to wait for Gemma before giving up on that alert

# What YOLO looks for. "orange" stays in on purpose: YOLO often calls a pumpkin "orange",
# so we catch it here and let CLIP make the real decision.
YOLO_WORDS = ["pumpkin", "jack-o'-lantern", "orange", "basketball", "ball", "traffic cone", "person"]
IGNORE_WORDS = ["person"]          # never sent to CLIP, never drawn

# What CLIP chooses between: (name shown on screen, description CLIP compares against)
CLIP_CHOICES = [
    ("pumpkin", "a photo of a pumpkin"),
    ("pumpkin", "a photo of a jack-o'-lantern"),
    ("pumpkin", "a photo of a small decorative pumpkin with a stem"),
    ("orange", "a photo of an orange, the citrus fruit"),
    ("basketball", "a photo of a basketball"),
    ("ball", "a photo of an orange ball"),
    ("traffic cone", "a photo of a traffic cone"),
    ("something else", "a photo of a person's face"),
    ("something else", "a photo of a room"),
]
# ====================================================================

WINDOW = "Pumpkin Cam (press Q to quit)"
GREEN, RED, GRAY, YELLOW = (0, 170, 0), (0, 0, 200), (90, 90, 90), (0, 200, 230)   # BGR

try:
    from notify import send_alert, send_gemma_verdict
    print("Discord/phone alerts: ON")
except Exception as e:
    print("Alerts to Discord/phone disabled:", e)
    send_alert = send_gemma_verdict = None


# ---------------------------  CLIP judge  ---------------------------
clip_model = clip_preprocess = clip_text = None


def load_clip():
    global clip_model, clip_preprocess, clip_text
    import torch
    import open_clip
    print("Loading CLIP judge...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        clip_text = clip_model.encode_text(tokenizer([desc for _, desc in CLIP_CHOICES]))
        clip_text /= clip_text.norm(dim=-1, keepdim=True)


def crop_box(frame, xyxy, pad=0.10):
    """Cut a box out of the frame, with a little extra border."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    return frame[y1:y2, x1:x2]


def clip_judge(frame, boxes_xyxy):
    """For each box: (pumpkin score 0-1, best non-pumpkin name). All boxes in one batch."""
    import torch
    from PIL import Image
    crops = [crop_box(frame, b) for b in boxes_xyxy]
    keep = [i for i, c in enumerate(crops) if c.size > 0]
    out = [(0.0, "something else")] * len(boxes_xyxy)
    if not keep:
        return out
    imgs = torch.stack([clip_preprocess(Image.fromarray(cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB)))
                        for i in keep])
    with torch.no_grad():
        feats = clip_model.encode_image(imgs)
        feats /= feats.norm(dim=-1, keepdim=True)
        probs = (100 * feats @ clip_text.T).softmax(dim=-1)
    for row, i in zip(probs.tolist(), keep):
        pumpkin = sum(p for (name, _), p in zip(CLIP_CHOICES, row) if name == "pumpkin")
        others = [(p, name) for (name, _), p in zip(CLIP_CHOICES, row) if name != "pumpkin"]
        out[i] = (pumpkin, max(others)[1])
    return out


# ---------------------------  Alerts  ---------------------------
def beep():
    if sys.platform == "darwin":
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
    else:
        print("\a", end="", flush=True)


GEMMA_PROMPT = ("Look at the main object in this photo. Is it a real pumpkin "
                "(including a carved jack-o'-lantern)? It might instead be an orange, a ball, "
                "or something else orange. Reply in JSON with: is_pumpkin (true/false), "
                "confidence (0-100, how sure you are), and reason (one short sentence about "
                "what you see, like the stem, ridges, size or texture).")
GEMMA_SCHEMA = {
    "type": "object",
    "properties": {
        "is_pumpkin": {"type": "boolean"},
        "confidence": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["is_pumpkin", "confidence", "reason"],
}


def gemma_judge(image_path):
    """Ask Gemma about one cropped photo. Returns dict: is_pumpkin, confidence, reason."""
    import json
    import ollama
    client = ollama.Client(timeout=GEMMA_TIMEOUT)
    reply = client.chat(model=GEMMA_MODEL, format=GEMMA_SCHEMA, options={"temperature": 0},
                        messages=[{"role": "user", "content": GEMMA_PROMPT, "images": [image_path]}])
    verdict = json.loads(reply["message"]["content"])
    verdict["confidence"] = max(0, min(100, int(verdict.get("confidence", 0))))
    return verdict


def gemma_warmup():
    """Load Gemma into memory at startup so the first alert isn't extra slow."""
    try:
        import ollama
        print("Loading Gemma 4 in the background...")
        ollama.chat(model=GEMMA_MODEL, messages=[{"role": "user", "content": "Say ok."}])
        print("Gemma 4 ready.")
    except Exception as e:
        print(f"Gemma not available ({e}). Is the Ollama app running? Did you run: ollama pull {GEMMA_MODEL}")


def ask_gemma(crop_path):
    """Returns (verdict dict or None, error text or None)."""
    start = time.time()
    try:
        verdict = gemma_judge(crop_path)
        word = "PUMPKIN" if verdict["is_pumpkin"] else "NOT a pumpkin"
        print(f"Gemma 4: {word}, {verdict['confidence']}% sure - {verdict['reason']}"
              f"  ({time.time() - start:.0f}s)")
        return verdict, None
    except Exception as e:
        print("Gemma check failed:", e)
        return None, f"Gemma 4 didn't answer ({type(e).__name__})."


def gemma_then_send(photo_path, crop_path, count):
    """Runs in the background so the video never freezes."""
    if GEMMA_VETO:
        # wait for Gemma first; only alert if it agrees
        verdict, _ = ask_gemma(crop_path)
        if verdict and not verdict["is_pumpkin"]:
            print("Gemma vetoed this alert, not sending.")
            return
        if send_alert:
            print("Sending alert to Discord...")
            send_alert(photo_path, count=count, gemma=verdict)
        return

    # default: snapshot card goes out right away, Gemma's verdict follows when ready
    if send_alert:
        print("Sending alert to Discord...")
        send_alert(photo_path, count=count, gemma_pending=True)
    verdict, error = ask_gemma(crop_path)
    if send_gemma_verdict:
        send_gemma_verdict(verdict, error, image_path=crop_path)


def fire_alert(image, count, frame=None, pumpkin_boxes=()):
    os.makedirs("alerts", exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join("alerts", stamp + ".jpg")
    cv2.imwrite(path, image)
    print(f"ALERT: {count} pumpkin(s)! Saved {path}")
    beep()

    crop_path = None
    if USE_GEMMA and frame is not None and pumpkin_boxes:
        biggest = max(pumpkin_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        crop = crop_box(frame, biggest)
        if crop.size > 0:
            crop_path = os.path.join("alerts", stamp + "-crop.jpg")
            cv2.imwrite(crop_path, crop)

    if crop_path or send_alert:
        # background thread so the video doesn't freeze while Gemma thinks / Discord uploads
        if crop_path is None:
            print("Sending alert to Discord...")
            threading.Thread(target=send_alert, args=(path, count), daemon=True).start()
        else:
            threading.Thread(target=gemma_then_send, args=(path, crop_path, count), daemon=True).start()


# ---------------------------  Drawing  ---------------------------
def draw_box(view, xyxy, label, color):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(view, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(view, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(view, label, (x1 + 4, max(th, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_banner(view, text, color, fps):
    h, w = view.shape[:2]
    bar = max(40, h // 9)
    scale = max(0.6, w / 700)
    thick = max(2, int(scale * 2))
    cv2.rectangle(view, (0, 0), (w, bar), color, -1)
    cv2.putText(view, text, (15, int(bar * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)
    cv2.putText(view, f"{fps:.1f} FPS", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def center(xyxy):
    return ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)


# ---------------------------  Main loop  ---------------------------
def main():
    print("Loading YOLO-World (10-30 seconds on first start)...")
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(YOLO_WORDS)
    if USE_CLIP:
        load_clip()
    if USE_GEMMA:
        threading.Thread(target=gemma_warmup, daemon=True).start()

    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        print("Could not open the camera. Close Zoom/FaceTime/Discord, or try CAMERA_INDEX = 1.")
        return
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Running! Hold up a pumpkin. Press Q in the video window to quit.")

    recent = deque(maxlen=FRAMES_TO_REMEMBER)
    verdicts = []            # CLIP memory: list of (center, pumpkin_score, other_name)
    confirmed = False
    last_seen = 0.0
    shown_count = 0
    last_alert = 0.0
    frame_num = 0
    bad_reads = 0
    fps, t_prev = 0.0, time.time()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                bad_reads += 1
                if bad_reads > 100:
                    print("Lost the camera feed.")
                    break
                time.sleep(0.05)
                continue
            bad_reads = 0
            frame_num += 1
            h, w = frame.shape[:2]
            near = 0.15 * math.hypot(w, h)   # how close a box must be to reuse an earlier CLIP verdict

            # 1. YOLO finds candidates
            result = model.predict(frame, conf=DETECT_CONF, imgsz=IMG_SIZE,
                                   agnostic_nms=True, verbose=False)[0]
            cands = []
            for b in result.boxes:
                name = result.names[int(b.cls)]
                if name in IGNORE_WORDS:
                    continue
                cands.append((float(b.conf), name, b.xyxy[0].tolist()))
            cands = sorted(cands, reverse=True)[:MAX_OBJECTS]

            # 2. Decide what each candidate is
            judged = []   # (xyxy, is_pumpkin, label)
            if USE_CLIP and cands:
                # reuse recent CLIP verdicts for objects that haven't moved much
                matched = []
                for conf, name, xyxy in cands:
                    cx, cy = center(xyxy)
                    best = min(verdicts, key=lambda v: math.hypot(v[0][0] - cx, v[0][1] - cy), default=None)
                    if best and math.hypot(best[0][0] - cx, best[0][1] - cy) < near:
                        matched.append(best)
                    else:
                        matched.append(None)

                if frame_num % CLIP_EVERY == 0 or None in matched:
                    fresh = clip_judge(frame, [x for _, _, x in cands])
                    new_verdicts = []
                    for (conf, name, xyxy), old, (score, other) in zip(cands, matched, fresh):
                        if old:                      # smooth with the previous score so it's steady
                            score = 0.5 * old[1] + 0.5 * score
                        new_verdicts.append((center(xyxy), score, other))
                    verdicts = new_verdicts
                else:
                    verdicts = [(center(x), m[1], m[2]) for (_, _, x), m in zip(cands, matched)]

                for (conf, name, xyxy), (_, score, other) in zip(cands, verdicts):
                    if score >= CLIP_MIN_SCORE:
                        judged.append((xyxy, True, f"PUMPKIN {score:.0%}"))
                    else:
                        judged.append((xyxy, False, f"{other} (pumpkin {score:.0%})"))
            else:
                verdicts = []
                for conf, name, xyxy in cands:
                    is_p = name in ("pumpkin", "jack-o'-lantern")
                    judged.append((xyxy, is_p, f"{name} {conf:.0%}"))

            pumpkins_now = sum(1 for _, is_p, _ in judged if is_p)
            rejects = [lbl.split(" (")[0] for _, is_p, lbl in judged if not is_p]

            # 3. Patience + hold: turn green after a few sightings, stay green for HOLD_SECONDS
            now = time.time()
            recent.append(pumpkins_now > 0)
            if pumpkins_now:
                last_seen = now
                shown_count = pumpkins_now
            if sum(recent) >= FRAMES_NEEDED:
                confirmed = True
            elif now - last_seen > HOLD_SECONDS:
                confirmed = False

            if confirmed:
                text = f"{shown_count} PUMPKINS DETECTED" if shown_count >= 2 else "PUMPKIN DETECTED"
                color = GREEN
            elif rejects:
                text, color = f"Not a pumpkin: {rejects[0]}", RED
            else:
                text, color = "No pumpkin", GRAY

            view = frame.copy()
            for xyxy, is_p, lbl in judged:
                draw_box(view, xyxy, lbl, GREEN if is_p else RED)

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            draw_banner(view, text, color, fps)

            # 4. Alert: snapshot = exactly what's on screen (boxes + green banner)
            if confirmed and pumpkins_now and now - last_alert > ALERT_COOLDOWN:
                last_alert = now
                fire_alert(view.copy(), shown_count, frame,
                           [xyxy for xyxy, is_p, _ in judged if is_p])

            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
