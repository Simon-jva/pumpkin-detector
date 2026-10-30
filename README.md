# Pumpkin Detector

A security camera that spots pumpkins and ignores look-alikes (basketballs, oranges, orange shirts).

## How it works

webcam → YOLO-World finds possible pumpkins → must stay in view ~1 second → CLIP double-checks the crop → alert (beep + saved photo + Discord / phone notification)

## Setup (one time)

- Mac/Linux: `bash setup_mac.sh`
- Windows: `setup_windows.bat`

## Alerts to Discord or your phone

Copy `.env.example` to `.env`, add a Discord webhook URL and/or an ntfy topic, then test with `python notify.py`.

## Run

```
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

python app.py
```

## Folders

- `test/pumpkin/` — photos of real pumpkins (for measuring accuracy)
- `test/not_pumpkin/` — tricky look-alikes (basketballs, oranges, cones)
- `alerts/` — snapshots saved when a pumpkin is detected

## Results

_Fill in after running `eval.py` during the hackathon._
