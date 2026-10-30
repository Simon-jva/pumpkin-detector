"""
Sends pumpkin alerts somewhere outside your laptop.

Two options (use one or both):
  1. Discord  - posts the alert + the snapshot photo into a Discord channel
  2. ntfy.sh  - free push notification to your phone, no account needed

Setup: copy .env.example to .env and fill in what you want to use.
Test:  python notify.py      (sends a test alert)

In app.py, after saving the snapshot:
    from notify import send_alert
    send_alert("alerts/20261030-101500.jpg")

Tip: call send_alert in a background thread so the video doesn't freeze:
    import threading
    threading.Thread(target=send_alert, args=(path,), daemon=True).start()
"""
import os
from datetime import datetime

import requests


def load_env(path=".env"):
    """Read KEY=value lines from .env (keeps your secret URLs out of GitHub)."""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


load_env()
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")


def send_discord(message, image_path=None):
    if not DISCORD_WEBHOOK_URL:
        return
    data = {"content": message}
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            r = requests.post(DISCORD_WEBHOOK_URL, data=data,
                              files={"file": (os.path.basename(image_path), f, "image/jpeg")},
                              timeout=10)
    else:
        r = requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=10)
    r.raise_for_status()
    print("Discord: sent")


def send_ntfy(message, image_path=None):
    if not NTFY_TOPIC:
        return
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    headers = {"Title": "Pumpkin detected!", "Tags": "jack_o_lantern"}
    if image_path and os.path.exists(image_path):
        headers["Filename"] = os.path.basename(image_path)
        headers["Message"] = message
        with open(image_path, "rb") as f:
            r = requests.put(url, data=f, headers=headers, timeout=10)
    else:
        r = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=10)
    r.raise_for_status()
    print("ntfy: sent")


def send_alert(image_path=None, confidence=None):
    """Call this when a pumpkin alert fires. Never crashes your app."""
    when = datetime.now().strftime("%I:%M:%S %p")
    message = f"Pumpkin detected at {when}"
    if confidence is not None:
        message += f" (confidence {confidence:.0%})"
    for sender in (send_discord, send_ntfy):
        try:
            sender(message, image_path)
        except Exception as e:
            print(f"{sender.__name__} failed: {e}")


if __name__ == "__main__":
    if not DISCORD_WEBHOOK_URL and not NTFY_TOPIC:
        print("Nothing set up yet. Copy .env.example to .env and fill in a Discord URL or ntfy topic.")
    else:
        send_alert(confidence=0.97)
        print("Test alert sent. Check Discord / your phone!")
