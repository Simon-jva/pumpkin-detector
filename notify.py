"""
Sends pumpkin alerts somewhere outside your laptop.

Two options (use one or both):
  1. Discord  - card 1: snapshot photo + time + pumpkin count (sent instantly)
                card 2: Gemma 4's verdict + "Gemma confidence: 93%" + reason
                        + the close-up photo Gemma looked at (sent when Gemma finishes)
  2. ntfy.sh  - free push notification to your phone, no account needed

Setup: copy .env.example to .env and fill in what you want to use.
Test:  python notify.py      (sends a test alert WITH a test photo)
"""
import os
import json
from datetime import datetime, timezone

import requests


def load_env(path=".env"):
    """Read KEY=value lines from .env (keeps your secret URLs out of GitHub)."""
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")

ORANGE, GREEN, RED = 0xFF7518, 0x2ECC71, 0xE74C3C


def gemma_text(gemma):
    """The Gemma lines shown in Discord."""
    verdict = "Pumpkin ✅" if gemma.get("is_pumpkin") else "Not a pumpkin ❌"
    return (f"**Gemma verdict:** {verdict}\n"
            f"**Gemma confidence:** {gemma.get('confidence', 0)}%\n"
            f"**Gemma says:** {gemma.get('reason', '-')[:1000]}")


def post_to_discord(embed, image_path=None, image_spot="image"):
    """Post one card. If image_path exists, the photo is uploaded and shown inside the card.
    image_spot = "image" (big photo) or "thumbnail" (small photo in the corner)."""
    has_photo = bool(image_path) and os.path.exists(image_path)
    if has_photo:
        name = "snapshot.jpg"
        embed[image_spot] = {"url": f"attachment://{name}"}
        payload = {"embeds": [embed], "attachments": [{"id": 0, "filename": name}]}
        with open(image_path, "rb") as f:
            r = requests.post(DISCORD_WEBHOOK_URL,
                              data={"payload_json": json.dumps(payload)},
                              files={"files[0]": (name, f, "image/jpeg")},
                              timeout=30)
    else:
        if image_path:
            print(f"Discord: photo not found at {image_path}, sending without it")
        r = requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=30)
    if r.status_code >= 400:
        print("Discord said:", r.status_code, r.text[:300])
    r.raise_for_status()
    return has_photo


def send_discord(image_path=None, count=None, gemma=None, gemma_pending=False):
    if not DISCORD_WEBHOOK_URL:
        return
    lines = [f"**Time:** {datetime.now().strftime('%I:%M:%S %p')}"]
    if count:
        lines.append(f"**Pumpkins seen:** {count}")
    if gemma is not None:
        lines.append(gemma_text(gemma))
        color = GREEN if gemma.get("is_pumpkin") else RED
    else:
        if gemma_pending:
            lines.append("**Gemma confidence:** checking the photo... (verdict coming next)")
        color = ORANGE
    embed = {
        "title": "🎃 Pumpkin detected!",
        "description": "\n".join(lines),
        "color": color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "Pumpkin Cam - YOLO-World + CLIP + Gemma 4"},
    }
    with_photo = post_to_discord(embed, image_path, "image")
    print("Discord: sent" + (" with photo" if with_photo else " (no photo)"))


def send_discord_gemma(gemma, error=None, image_path=None):
    """Second card with Gemma's verdict and the close-up photo Gemma looked at."""
    if not DISCORD_WEBHOOK_URL:
        return
    if gemma is None:
        embed = {"title": "🤖 Gemma 4 couldn't check this one",
                 "description": error or "Gemma was offline or took too long.", "color": ORANGE}
    else:
        embed = {"title": "🤖 Gemma 4's verdict", "description": gemma_text(gemma),
                 "color": GREEN if gemma.get("is_pumpkin") else RED}
    post_to_discord(embed, image_path, "thumbnail")
    print("Discord: Gemma verdict sent")


def send_ntfy(image_path=None, count=None, gemma=None, gemma_pending=False):
    if not NTFY_TOPIC:
        return
    message = f"Pumpkin detected at {datetime.now().strftime('%I:%M:%S %p')}"
    if gemma:
        word = "pumpkin" if gemma.get("is_pumpkin") else "NOT a pumpkin"
        message += f" - Gemma: {word}, Gemma confidence: {gemma.get('confidence', 0)}%"
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    headers = {"Title": "Pumpkin detected!", "Tags": "jack_o_lantern"}
    if image_path and os.path.exists(image_path):
        headers["Filename"] = "snapshot.jpg"
        headers["Message"] = message
        with open(image_path, "rb") as f:
            r = requests.put(url, data=f, headers=headers, timeout=30)
    else:
        r = requests.post(url, data=message.encode("utf-8"), headers=headers, timeout=30)
    r.raise_for_status()
    print("ntfy: sent")


def send_alert(image_path=None, count=None, gemma=None, gemma_pending=False):
    """Call this when a pumpkin alert fires. Never crashes your app.

    image_path:    snapshot photo to attach
    count:         how many pumpkins were seen
    gemma:         dict from Gemma (is_pumpkin, confidence, reason), or None
    gemma_pending: True = Gemma is still thinking; its verdict follows via send_gemma_verdict
    """
    for sender in (send_discord, send_ntfy):
        try:
            sender(image_path, count, gemma, gemma_pending)
        except Exception as e:
            print(f"{sender.__name__} failed: {e}")


def send_gemma_verdict(gemma, error=None, image_path=None):
    """Follow-up message once Gemma has answered. Never crashes your app."""
    try:
        send_discord_gemma(gemma, error, image_path)
    except Exception as e:
        print(f"send_discord_gemma failed: {e}")


def make_test_photo(path="alerts/test-snapshot.jpg"):
    """Draws a simple orange 'pumpkin' picture so the test has a photo to send."""
    from PIL import Image, ImageDraw
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (640, 480), (30, 30, 40))
    d = ImageDraw.Draw(img)
    d.ellipse((200, 150, 440, 380), fill=(235, 120, 20))
    d.rectangle((310, 115, 330, 160), fill=(60, 120, 40))
    d.rectangle((0, 0, 640, 50), fill=(0, 170, 0))
    d.text((15, 18), "TEST SNAPSHOT - PUMPKIN DETECTED", fill=(255, 255, 255))
    img.save(path, "JPEG")
    return path


if __name__ == "__main__":
    if not DISCORD_WEBHOOK_URL and not NTFY_TOPIC:
        print("Nothing set up yet. Copy .env.example to .env and fill in a Discord URL or ntfy topic.")
    else:
        photo = make_test_photo()
        send_alert(photo, count=1, gemma_pending=True)
        send_gemma_verdict({"is_pumpkin": True, "confidence": 95,
                            "reason": "TEST - orange ribbed skin with a green stem."}, image_path=photo)
        print("Test sent: 2 cards, both with photos. Check Discord!")
