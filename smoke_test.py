"""
Smoke test: checks that everything is installed and downloads the AI models
so they're saved on your laptop BEFORE the hackathon (no Wi-Fi needed later).

Run:  python smoke_test.py
Each step prints PASS or FAIL. Fix any FAIL tonight.
"""
import sys

results = []


def step(name, fn):
    print(f"\n--- {name} ---")
    try:
        fn()
        print(f"PASS: {name}")
        results.append((name, True))
    except Exception as e:  # show the error but keep going
        print(f"FAIL: {name}\n   {type(e).__name__}: {e}")
        results.append((name, False))


def check_python():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if not (v.major == 3 and 10 <= v.minor <= 12):
        raise RuntimeError("Use Python 3.10, 3.11 or 3.12 (newer versions can break PyTorch).")


def check_torch():
    import torch
    print("PyTorch", torch.__version__)
    if torch.cuda.is_available():
        print("NVIDIA GPU found:", torch.cuda.get_device_name(0), "(fast!)")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) found (fast!)")
    else:
        print("No GPU found, will use CPU (fine, just slower; keep image size small)")


def check_yolo_world():
    import numpy as np
    from ultralytics import YOLOWorld
    model = YOLOWorld("yolov8s-worldv2.pt")  # downloads ~25 MB the first time
    model.set_classes(["pumpkin", "basketball", "orange"])  # may install a CLIP helper the first time
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(blank, verbose=False)
    print("YOLO-World loaded and ran on a test image")


def check_clip():
    import open_clip
    import torch
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )  # downloads ~350 MB the first time
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    img = preprocess(Image.new("RGB", (224, 224), (255, 140, 0))).unsqueeze(0)
    text = tokenizer(["a photo of a pumpkin", "a photo of a basketball"])
    with torch.no_grad():
        probs = (model.encode_image(img) @ model.encode_text(text).T).softmax(dim=-1)
    print("CLIP loaded; test scores:", probs.tolist())


def check_webcam():
    import cv2
    cam = cv2.VideoCapture(0)
    ok, frame = cam.read()
    cam.release()
    if not ok:
        raise RuntimeError(
            "Could not read from webcam. On Mac: System Settings > Privacy & Security > "
            "Camera, allow Terminal (or VS Code). Close Zoom/Teams if they're using it."
        )
    print("Webcam works, frame size:", frame.shape)


step("Python version", check_python)
step("PyTorch", check_torch)
step("YOLO-World (pumpkin detector)", check_yolo_world)
step("CLIP (double-checker)", check_clip)
step("Webcam", check_webcam)

print("\n========== SUMMARY ==========")
for name, ok in results:
    print(("PASS  " if ok else "FAIL  ") + name)
if all(ok for _, ok in results):
    print("\nEverything works. You're ready for the hackathon!")
else:
    print("\nFix the FAIL lines above (the PDF guide has a troubleshooting section).")
