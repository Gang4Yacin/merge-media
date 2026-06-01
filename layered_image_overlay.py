"""Layered ecommerce image generator (ordered layer model).

Unlike custom_image_overlay.py (which runs all photos, then all texts, in two
fixed phases), this script composites an *ordered* list of layers. The order of
the list is the z-order: each layer is drawn on top of the current image, so a
chroma_photo placed after an overlay targets whatever is visible at that point.

Layer types
-----------
- background   {url}
      Downloads the base image and establishes the canvas. Must be the first
      layer. A later "background" is treated as a full-bleed overlay.
- chroma_photo {url, color, tolerance?}
      Locates the solid color zone *in the current composite*, fits the photo
      "cover" (no distortion), clips it to the zone's real shape, and pastes it.
- overlay      {url, fit?, opacity?}
      Draws an image full-bleed over the whole canvas. Honors PNG transparency.
      fit: "cover" (default) | "contain" | "stretch". opacity: 0..1 (default 1).
- image        {url, x, y, w, h, shape?, opacity?}
      Places an image at (x, y) sized (w, h), masked to a shape. shape:
      "square" (default) | "circle". Source transparency preserved.
- text         {text_b64, x, y, w, h, color?, bold?, font?, font_size?, autoshrink?}
      Renders text with Playwright (word-wrap, color emoji) and composites it
      at (x, y). color (default #000000), bold (default false), font ("Rubik"
      default, or "Quicksand"), font_size (default 19). autoshrink (default
      true) shrinks the text to fit the box; false applies font_size exactly.

Input JSON — any of:
  { "layers_json": "<json string of a layer array>" }
  { "layers": [ ... ] }
  [ {type:...}, ... ]                      # a single image
  [ { "layers": [ ... ] }, ... ]           # multiple images
Each image's final PNG is uploaded to the Supabase "growth-marketing" bucket.
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

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
# Available text fonts (name -> file). Rubik is a variable font, so a single
# file covers every weight (font-weight in CSS selects the instance).
FONTS = {
    "Rubik": "Rubik-VariableFont_wght.ttf",
    "Quicksand": "Quicksand-Regular.ttf",
}
DEFAULT_FONT = "Rubik"
# Apple Color Emoji (iOS-style glyphs). Downloaded by the CI workflow.
EMOJI_FONT_PATH = os.path.join(FONT_DIR, "AppleColorEmoji.ttf")

SUPABASE_BUCKET = "growth-marketing"
SUPABASE_HOST = "https://bksiaeiqzmoaxvkdtspn.supabase.co"
SUPABASE_UPLOAD_URL = f"{SUPABASE_HOST}/storage/v1/object/{SUPABASE_BUCKET}"
SUPABASE_PUBLIC_PREFIX = (
    f"{SUPABASE_HOST}/storage/v1/object/public/{SUPABASE_BUCKET}")

DEFAULT_FONT_SIZE = 19
DEFAULT_COLOR_TOLERANCE = 12
MIN_FONT_SIZE = 6
DEFAULT_TEXT_COLOR = "#000000"
LINE_HEIGHT = 1.3


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def download_image(url, keep_alpha=False):
    """Download an image. Returns (image, error_message).

    keep_alpha=True returns BGRA when the source has an alpha channel.
    """
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
        if not keep_alpha:
            img = to_bgr(img)
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


def to_bgra(img):
    """Ensure the image is 4-channel BGRA (opaque alpha if none)."""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
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


def fit_cover(img, box_w, box_h):
    """Scale-to-fill + center-crop img to exactly box_w x box_h (no distortion).

    Preserves the channel count (BGR or BGRA).
    """
    ih, iw = img.shape[:2]
    scale = max(box_w / iw, box_h / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    x0 = (new_w - box_w) // 2
    y0 = (new_h - box_h) // 2
    return resized[y0:y0 + box_h, x0:x0 + box_w]


def fit_contain(img, box_w, box_h):
    """Scale to fit inside the box (no crop). Returns (resized, off_x, off_y)."""
    ih, iw = img.shape[:2]
    scale = min(box_w / iw, box_h / ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    off_x = (box_w - new_w) // 2
    off_y = (box_h - new_h) // 2
    return resized, off_x, off_y


def alpha_composite(bg, overlay_bgra, x, y, opacity=1.0):
    """Alpha-composite a BGRA overlay onto a BGR bg at (x, y). Mutates bg."""
    bg_h, bg_w = bg.shape[:2]
    oh, ow = overlay_bgra.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(bg_w, x + ow), min(bg_h, y + oh)
    if x1 <= x0 or y1 <= y0:
        return
    ov = overlay_bgra[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = (ov[:, :, 3:4].astype(np.float32) / 255.0) * float(opacity)
    roi = bg[y0:y1, x0:x1].astype(np.float32)
    blended = ov[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    bg[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Layer: chroma_photo
# --------------------------------------------------------------------------- #
def find_color_zone(bg, color, tolerance):
    """Locate the largest region matching color in the current composite.

    Returns ((x, y, w, h), zone_mask) or (None, error_str).
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
    zone_mask = np.zeros(bg.shape[:2], dtype=np.uint8)
    cv2.drawContours(zone_mask, [largest], -1, 255, thickness=cv2.FILLED)
    zone_mask = cv2.bitwise_or(zone_mask, cv2.bitwise_and(mask, zone_mask))
    return (x, y, w, h), zone_mask


def apply_chroma_photo(bg, layer):
    """Fill a color zone (detected on the current composite) with a photo."""
    url = layer.get("url")
    color = layer.get("color")
    if not url or not color:
        return "chroma_photo: missing url or color"
    tol = int(layer.get("tolerance", DEFAULT_COLOR_TOLERANCE))
    zone, zone_mask = find_color_zone(bg, color, tol)
    if zone is None:
        return zone_mask  # error string
    x, y, w, h = zone
    img, derr = download_image(url)
    if img is None:
        return f"chroma_photo: failed to download photo: {derr}"
    filled = fit_cover(img, w, h)
    roi_mask = zone_mask[y:y + h, x:x + w].astype(np.float32) / 255.0
    roi_mask = cv2.GaussianBlur(roi_mask, (0, 0), sigmaX=0.8)
    a = roi_mask[:, :, None]
    roi = bg[y:y + h, x:x + w].astype(np.float32)
    bg[y:y + h, x:x + w] = np.clip(
        filled.astype(np.float32) * a + roi * (1.0 - a),
        0, 255).astype(np.uint8)
    return None


# --------------------------------------------------------------------------- #
# Layer: overlay (full-bleed)
# --------------------------------------------------------------------------- #
def apply_overlay(bg, layer):
    """Draw an image full-bleed over the whole canvas (alpha honored)."""
    url = layer.get("url")
    if not url:
        return "overlay: missing url"
    fit = (layer.get("fit") or "cover").lower()
    opacity = float(layer.get("opacity", 1.0))
    img, derr = download_image(url, keep_alpha=True)
    if img is None:
        return f"overlay: failed to download image: {derr}"
    img = to_bgra(img)
    bg_h, bg_w = bg.shape[:2]
    if fit == "stretch":
        resized = cv2.resize(img, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
        alpha_composite(bg, resized, 0, 0, opacity)
    elif fit == "contain":
        resized, off_x, off_y = fit_contain(img, bg_w, bg_h)
        alpha_composite(bg, resized, off_x, off_y, opacity)
    else:  # cover (default)
        resized = fit_cover(img, bg_w, bg_h)
        alpha_composite(bg, resized, 0, 0, opacity)
    return None


# --------------------------------------------------------------------------- #
# Layer: image (free placement at x/y, sized w/h, square or circle)
# --------------------------------------------------------------------------- #
def apply_image(bg, layer):
    """Place an image at (x, y) sized (w, h), masked to a shape.

    shape: "square" (default, rectangle) | "circle" (ellipse inscribed in the
    box; a true circle when w == h). The source image's own transparency is
    preserved and combined with the shape mask. opacity (0..1) optional.
    """
    url = layer.get("url")
    if not url:
        return "image: missing url"
    try:
        x, y = int(layer["x"]), int(layer["y"])
        w, h = int(layer["w"]), int(layer["h"])
    except (KeyError, ValueError, TypeError) as e:
        return f"image: bad coordinates: {e}"
    if w <= 0 or h <= 0:
        return "image: w and h must be positive"
    shape = (layer.get("shape") or "square").lower()
    opacity = float(layer.get("opacity", 1.0))
    img, derr = download_image(url, keep_alpha=True)
    if img is None:
        return f"image: failed to download: {derr}"
    img = fit_cover(to_bgra(img), w, h)  # exactly (h, w), 4 channels
    if shape in ("circle", "ellipse", "round"):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)),
                    0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.8)
        src_a = img[:, :, 3].astype(np.float32)
        img[:, :, 3] = (src_a * (mask.astype(np.float32) / 255.0)).astype(np.uint8)
    alpha_composite(bg, img, x, y, opacity)
    return None


# --------------------------------------------------------------------------- #
# Layer: text
# --------------------------------------------------------------------------- #
def _font_data_uri(font_name):
    fname = FONTS.get(font_name, FONTS[DEFAULT_FONT])
    with open(os.path.join(FONT_DIR, fname), "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:font/ttf;base64,{b64}"


def _emoji_face_css():
    """@font-face for Apple Color Emoji, or '' if the font is unavailable."""
    if not os.path.exists(EMOJI_FONT_PATH):
        return ""
    with open(EMOJI_FONT_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return ("@font-face { font-family:'Apple Color Emoji'; "
            "src:url('data:font/ttf;base64,%s'); }" % b64)


def render_text_png(page, text, box_w, box_h, font_size, color, bold,
                    font_name, autoshrink=True):
    """Render text in a box_w x box_h transparent PNG.

    font_size is the requested size. When autoshrink is True (default), the
    size is reduced until the text fits the box; when False, font_size is
    applied exactly (the text may overflow the box).
    """
    font_uri = _font_data_uri(font_name)
    weight = 700 if bold else 400
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  @font-face {{ font-family:'UserFont'; src:url('{font_uri}') format('truetype');
               font-weight: 100 900; }}
  {emoji_face}
  html,body {{ margin:0; padding:0; background:transparent; }}
  #wrap {{ width:{w}px; height:{h}px; display:flex; align-items:center;
           box-sizing:border-box; }}
  #txt {{ width:100%; font-family:'UserFont', 'Apple Color Emoji', sans-serif;
          font-weight:{weight}; color:{color}; text-align:left;
          white-space:pre-wrap; overflow-wrap:break-word; word-break:break-word;
          line-height:{lh}; }}
</style></head><body>
<div id="wrap"><div id="txt"></div></div>
</body></html>""".format(font_uri=font_uri, emoji_face=_emoji_face_css(),
                          w=box_w, h=box_h, color=color, weight=weight,
                          lh=LINE_HEIGHT)

    page.set_viewport_size({"width": box_w, "height": box_h})
    page.set_content(html, wait_until="load")
    page.evaluate("async () => { await document.fonts.ready; }")
    page.evaluate("(t) => { document.getElementById('txt').textContent = t; }",
                  text)
    if autoshrink:
        # Shrink the font until the text fits the box (centered in the box).
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
    else:
        # Apply the exact font size; let the box grow downward so the text is
        # never clipped (top-left anchored, width still wraps at box_w).
        page.evaluate(
            """(fs) => {
                const wrap = document.getElementById('wrap');
                const txt = document.getElementById('txt');
                txt.style.fontSize = fs + 'px';
                wrap.style.height = 'auto';
                wrap.style.alignItems = 'flex-start';
            }""",
            int(font_size),
        )
        dims = page.evaluate(
            """() => {
                const txt = document.getElementById('txt');
                return {w: Math.ceil(txt.scrollWidth),
                        h: Math.ceil(txt.scrollHeight)};
            }""")
        page.set_viewport_size({"width": max(box_w, dims["w"]),
                                "height": max(box_h, dims["h"])})
    return page.screenshot(omit_background=True)


def apply_text(bg, layer, page):
    """Render and composite a text layer at (x, y)."""
    raw = layer.get("text_b64")
    if raw is not None:
        try:
            text = base64.b64decode(raw).decode("utf-8")
        except Exception as e:
            return f"text: bad text_b64: {e}"
    else:
        text = str(layer.get("text", ""))
    try:
        x, y = int(layer["x"]), int(layer["y"])
        w, h = int(layer["w"]), int(layer["h"])
    except (KeyError, ValueError, TypeError) as e:
        return f"text: bad coordinates: {e}"
    color = layer.get("color") or DEFAULT_TEXT_COLOR
    bold = bool(layer.get("bold", False))
    font_name = layer.get("font") or DEFAULT_FONT
    font_size = layer.get("font_size") or DEFAULT_FONT_SIZE
    autoshrink = bool(layer.get("autoshrink", True))
    png = render_text_png(page, text, w, h, font_size, color, bold, font_name,
                          autoshrink)
    overlay = cv2.imdecode(np.frombuffer(png, dtype=np.uint8),
                           cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return "text: rendered PNG could not be decoded"
    alpha_composite(bg, to_bgra(overlay), x, y)
    return None


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Item processing
# --------------------------------------------------------------------------- #
def process_item(page, layers, idx, errors, token):
    """Composite an ordered list of layers. Returns (public_url, error)."""
    if not layers:
        return None, "No layers provided"

    # The first background layer establishes the canvas.
    bg = None
    for l_i, layer in enumerate(layers):
        ltype = (layer.get("type") or "").lower()

        if ltype == "background":
            url = layer.get("url")
            if not url:
                errors.append({"index": idx, "layer": l_i,
                               "error": "background: missing url"})
                continue
            img, derr = download_image(url)
            if img is None:
                return None, f"Failed to download background: {derr}"
            if bg is None:
                bg = img  # establishes the canvas
            else:
                err = apply_overlay(bg, {**layer, "fit": layer.get("fit",
                                                                   "cover")})
                if err:
                    errors.append({"index": idx, "layer": l_i, "error": err})
            continue

        if bg is None:
            return None, ("First layer must be a 'background' "
                          f"(got '{ltype}' at index {l_i})")

        if ltype == "chroma_photo":
            err = apply_chroma_photo(bg, layer)
        elif ltype == "overlay":
            err = apply_overlay(bg, layer)
        elif ltype == "image":
            err = apply_image(bg, layer)
        elif ltype == "text":
            err = apply_text(bg, layer, page)
        else:
            err = f"Unknown layer type: {ltype!r}"
        if err:
            errors.append({"index": idx, "layer": l_i, "error": err})

    if bg is None:
        return None, "No background layer found"
    ok, buf = cv2.imencode(".png", bg)
    if not ok:
        return None, "Failed to encode final PNG"
    return upload_to_supabase(buf.tobytes(), token)


def normalize_items(data):
    """Normalize any accepted input shape into a list of layer-lists."""
    if isinstance(data, dict):
        if "layers_json" in data:
            data = data["layers_json"]
            if isinstance(data, str):
                data = json.loads(data)
        elif "layers" in data:
            return [data["layers"]]
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "layers" in data:
        return [data["layers"]]
    if not isinstance(data, list):
        raise ValueError(f"expected a list, got {type(data)}")
    if not data:
        return []
    # List of items {layers:[...]} vs a bare list of layer dicts.
    if all(isinstance(x, dict) and "layers" in x for x in data):
        return [x["layers"] for x in data]
    return [data]


def main():
    parser = argparse.ArgumentParser(
        description="Composite an ordered list of image/text layers.")
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
        items = normalize_items(json.loads(content))
    except Exception as e:
        print(f"Error parsing JSON input: {e}")
        sys.exit(1)

    results, errors = [], []

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        for idx, layers in enumerate(items):
            print(f"\n--- Processing item {idx + 1}/{len(items)} "
                  f"({len(layers)} layers) ---")
            try:
                url, err = process_item(page, layers, idx, errors, token)
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
