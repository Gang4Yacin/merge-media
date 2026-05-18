"""Custom ecommerce image generator.

For each input item:
  1. Download a background template image.
  2. For each {url, color} photo: locate the solid color zone in the template,
     fit the photo "cover" (no distortion) centered, and paste it over the zone.
  3. For each {text_b64, x, y, w, h} text zone: render the text with Playwright
     (Quicksand font, color emoji, auto word-wrap, auto-shrink to fit the box)
     and alpha-composite it onto the image at (x, y).
  4. Upload the final PNG to the Supabase "growth-marketing" bucket and
     return its public URL (no callback to n8n, no DB insert).

Input JSON (list of items):
[
  {
    "bg_image": "https://...",
    "photos": [ { "url": "https://...", "color": "#FF00FF", "tolerance": 12 } ],
    "texts":  [ { "text_b64": "<base64 utf-8>", "x": 320, "y": 850,
                   "w": 540, "h": 180 } ],
    "font_size": 19
  }
]
"""

import argparse
import base64
import json
import os
import sys
import uuid

import cv2
import numpy as np
import requests

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fonts", "Quicksand-Regular.ttf")

SUPABASE_BUCKET = "growth-marketing"
SUPABASE_HOST = "https://bksiaeiqzmoaxvkdtspn.supabase.co"
SUPABASE_UPLOAD_URL = f"{SUPABASE_HOST}/storage/v1/object/{SUPABASE_BUCKET}"
SUPABASE_PUBLIC_PREFIX = (
    f"{SUPABASE_HOST}/storage/v1/object/public/{SUPABASE_BUCKET}")

DEFAULT_FONT_SIZE = 19
DEFAULT_COLOR_TOLERANCE = 12
MIN_FONT_SIZE = 6
TEXT_COLOR = "#000000"          # fixed for now (black)
LINE_HEIGHT = 1.3


def download_image(url):
    """Download an image. Returns (bgr_or_bgra_image, error_message)."""
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        arr = np.asarray(bytearray(r.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, ("Downloaded data is not a valid image "
                          f"(Content-Type: {r.headers.get('Content-Type')}, "
                          f"{len(r.content)} bytes)")
        return img, None
    except requests.exceptions.Timeout:
        return None, "Request timed out after 30s"
    except Exception as e:
        return None, f"Download error: {e}"


def to_bgr(img):
    """Drop alpha / expand grayscale so the image is 3-channel BGR."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def hex_to_bgr(color):
    """'#RRGGBB' or 'RRGGBB' -> (B, G, R) ints."""
    c = color.strip().lstrip("#")
    if len(c) != 6:
        raise ValueError(f"Invalid hex color: {color!r}")
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    return (b, g, r)


def find_color_zone(bg, color, tolerance):
    """Return (x, y, w, h) bounding box of the largest region matching color.

    Returns (None, error) if no zone is found.
    """
    b, g, r = hex_to_bgr(color)
    lower = np.array([max(0, b - tolerance),
                      max(0, g - tolerance),
                      max(0, r - tolerance)], dtype=np.uint8)
    upper = np.array([min(255, b + tolerance),
                      min(255, g + tolerance),
                      min(255, r + tolerance)], dtype=np.uint8)
    mask = cv2.inRange(bg, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, f"No zone found for color {color}"
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:
        return None, f"Color zone for {color} is too small"
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h), None


def fit_cover(img, box_w, box_h):
    """Scale-to-fill + center-crop img to exactly box_w x box_h (no distortion)."""
    ih, iw = img.shape[:2]
    scale = max(box_w / iw, box_h / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    x0 = (new_w - box_w) // 2
    y0 = (new_h - box_h) // 2
    return resized[y0:y0 + box_h, x0:x0 + box_w]


def place_photos(bg, photos, errors, idx):
    """Paste each photo into its color zone. Mutates bg in place."""
    for p_i, photo in enumerate(photos or []):
        url = photo.get("url")
        color = photo.get("color")
        if not url or not color:
            errors.append({"index": idx, "photo": p_i,
                            "error": "Missing url or color"})
            continue
        tol = int(photo.get("tolerance", DEFAULT_COLOR_TOLERANCE))
        zone, err = find_color_zone(bg, color, tol)
        if zone is None:
            errors.append({"index": idx, "photo": p_i, "error": err})
            continue
        x, y, w, h = zone
        img, derr = download_image(url)
        if img is None:
            errors.append({"index": idx, "photo": p_i,
                           "error": f"Failed to download photo: {derr}"})
            continue
        img = to_bgr(img)
        bg[y:y + h, x:x + w] = fit_cover(img, w, h)


def _font_data_uri():
    with open(FONT_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:font/ttf;base64,{b64}"


def render_text_png(page, text, box_w, box_h, font_size):
    """Render text in a box_w x box_h transparent PNG with auto-shrink.

    Returns PNG bytes (BGRA-decodable).
    """
    font_uri = _font_data_uri()
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  @font-face {{ font-family:'Quicksand'; src:url('{font_uri}') format('truetype'); }}
  html,body {{ margin:0; padding:0; background:transparent; }}
  #wrap {{ width:{w}px; height:{h}px; display:flex; align-items:center;
           box-sizing:border-box; }}
  #txt {{ width:100%; font-family:'Quicksand', sans-serif; color:{color};
          text-align:left; white-space:pre-wrap; overflow-wrap:break-word;
          word-break:break-word; line-height:{lh}; }}
</style></head><body>
<div id="wrap"><div id="txt"></div></div>
</body></html>""".format(font_uri=font_uri, w=box_w, h=box_h,
                          color=TEXT_COLOR, lh=LINE_HEIGHT)

    page.set_viewport_size({"width": box_w, "height": box_h})
    page.set_content(html, wait_until="load")
    page.evaluate("async () => { await document.fonts.ready; }")
    # Inject text safely as textContent (no HTML injection, keeps \n via pre-wrap)
    page.evaluate("(t) => { document.getElementById('txt').textContent = t; }",
                  text)
    # Auto-shrink until it fits the box (height and width)
    page.evaluate(
        """({fs, minFs}) => {
            const wrap = document.getElementById('wrap');
            const txt = document.getElementById('txt');
            let size = fs;
            txt.style.fontSize = size + 'px';
            while (size > minFs &&
                   (txt.scrollHeight > wrap.clientHeight ||
                    txt.scrollWidth  > wrap.clientWidth)) {
                size -= 1;
                txt.style.fontSize = size + 'px';
            }
        }""",
        {"fs": int(font_size), "minFs": MIN_FONT_SIZE},
    )
    return page.screenshot(omit_background=True)


def composite_text(bg, png_bytes, x, y):
    """Alpha-composite a BGRA PNG onto bg (BGR) at (x, y). Mutates bg."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    overlay = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return "Rendered text PNG could not be decoded"
    if overlay.shape[2] == 3:  # no alpha -> opaque paste
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    bg_h, bg_w = bg.shape[:2]
    oh, ow = overlay.shape[:2]
    # Clip to background bounds
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(bg_w, x + ow), min(bg_h, y + oh)
    if x1 <= x0 or y1 <= y0:
        return f"Text zone ({x},{y}) is outside the image"

    ov = overlay[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = (ov[:, :, 3:4].astype(np.float32)) / 255.0
    roi = bg[y0:y1, x0:x1].astype(np.float32)
    blended = ov[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    bg[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return None


def upload_to_supabase(png_bytes, token):
    """Upload PNG bytes to the growth-marketing bucket. Returns (url, error)."""
    filename = f"images/custom_{uuid.uuid4().hex}.png"
    url = f"{SUPABASE_UPLOAD_URL}/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "image/png",
        "x-upsert": "true",
    }
    try:
        r = requests.post(url, headers=headers, data=png_bytes, timeout=60)
        if r.status_code in (200, 201):
            return f"{SUPABASE_PUBLIC_PREFIX}/{filename}", None
        return None, f"Supabase upload HTTP {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return None, f"Supabase upload error: {e}"


def process_item(page, item, idx, errors, token):
    """Returns (public_url, error_message)."""
    bg_url = item.get("bg_image")
    if not bg_url:
        return None, "Missing bg_image"

    bg_raw, derr = download_image(bg_url)
    if bg_raw is None:
        return None, f"Failed to download bg_image: {derr}"
    bg = to_bgr(bg_raw)

    place_photos(bg, item.get("photos"), errors, idx)

    font_size = item.get("font_size") or DEFAULT_FONT_SIZE
    for t_i, t in enumerate(item.get("texts") or []):
        try:
            text = base64.b64decode(t["text_b64"]).decode("utf-8")
        except Exception as e:
            errors.append({"index": idx, "text": t_i,
                           "error": f"Bad text_b64: {e}"})
            continue
        try:
            x, y = int(t["x"]), int(t["y"])
            w, h = int(t["w"]), int(t["h"])
        except (KeyError, ValueError, TypeError) as e:
            errors.append({"index": idx, "text": t_i,
                           "error": f"Bad text coordinates: {e}"})
            continue
        png = render_text_png(page, text, w, h, font_size)
        cerr = composite_text(bg, png, x, y)
        if cerr:
            errors.append({"index": idx, "text": t_i, "error": cerr})

    ok, buf = cv2.imencode(".png", bg)
    if not ok:
        return None, "Failed to encode final PNG"
    return upload_to_supabase(buf.tobytes(), token)


def main():
    parser = argparse.ArgumentParser(
        description="Fill color zones with photos + overlay custom text.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    token = os.environ.get("SUPABASE_TOKEN")
    if not token:
        print("Error: SUPABASE_TOKEN environment variable is not set.")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read().strip()
    print(f"Raw input (first 120 chars): {content[:120]}")
    try:
        items = json.loads(content)
        if isinstance(items, dict) and "images_json" in items:
            items = items["images_json"]
        if isinstance(items, str):
            items = json.loads(items)
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            print(f"Error: expected a list, got {type(items)}")
            sys.exit(1)
    except Exception as e:
        print(f"Error parsing JSON input: {e}")
        sys.exit(1)

    results, errors = [], []

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        for idx, item in enumerate(items):
            print(f"\n--- Processing item {idx + 1}/{len(items)} ---")
            try:
                url, err = process_item(page, item, idx, errors, token)
            except Exception as e:
                url, err = None, f"Unhandled error: {e}"
            if url:
                results.append({"index": idx, "final_image": url})
                print(f"  OK -> {url}")
            else:
                errors.append({"index": idx, "error": err})
                print(f"  FAILED: {err}")
        browser.close()

    summary = f"{len(results)}/{len(items)} succeeded, {len(errors)} errors"
    out = {
        "results": results,
        "final_image": results[0]["final_image"] if results else None,
        "errors": errors,
        "summary": summary,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\n{summary}. Output saved to {args.output}")
    if errors:
        for e in errors:
            print(f"  ERROR {e}")


if __name__ == "__main__":
    main()
