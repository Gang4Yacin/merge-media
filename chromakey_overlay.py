import cv2
import numpy as np
import requests
import argparse
import sys
import json
import os
import uuid
import math

SUPABASE_URL = "https://bksiaeiqzmoaxvkdtspn.supabase.co/storage/v1/object/n8n-image-generation/images"
SUPABASE_PUBLIC_URL_PREFIX = "https://bksiaeiqzmoaxvkdtspn.supabase.co/storage/v1/object/public/n8n-image-generation/images"

# Chromakey tuning parameters
MIN_DIFF = 10.0      # Below this green-difference = fully opaque (foreground)
MAX_DIFF = 75.0      # Above this green-difference = fully transparent (green screen)
TARGET_HUE = 60.0    # HSV hue for #00FF00 green
HUE_HARD = 15.0      # Hard hue tolerance (degrees)
HUE_SOFT = 10.0      # Soft falloff beyond hard tolerance
SPILL_OFFSET = 5.0   # Green spill suppression offset (default, overridable per item)

# Target aspect ratio
TARGET_RATIO_W = 9
TARGET_RATIO_H = 16


def download_image(url):
    """Download an image from URL. Returns (image, error_message) tuple."""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, f"Downloaded data is not a valid image (Content-Type: {response.headers.get('Content-Type', 'unknown')}, size: {len(response.content)} bytes)"
            return img, None
        else:
            body_preview = response.text[:300]
            return None, f"HTTP {response.status_code}: {body_preview}"
    except requests.exceptions.Timeout:
        return None, "Request timed out after 30s"
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def upload_to_supabase(filepath, token, filename):
    url = f"{SUPABASE_URL}/{filename}"
    content_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": content_type
    }
    with open(filepath, "rb") as f:
        response = requests.post(url, headers=headers, data=f)

    if response.status_code in [200, 201]:
        return f"{SUPABASE_PUBLIC_URL_PREFIX}/{filename}"
    else:
        print(f"Failed to upload {filename} to Supabase: {response.text}")
        return None


def detect_aspect_ratio(width, height):
    """Detect the closest standard aspect ratio for given dimensions."""
    STANDARD_RATIOS = [
        (1, 1, "1:1"),
        (4, 5, "4:5"),
        (5, 4, "5:4"),
        (9, 16, "9:16"),
        (16, 9, "16:9"),
        (2, 3, "2:3"),
        (3, 2, "3:2"),
        (3, 4, "3:4"),
        (4, 3, "4:3"),
    ]
    ratio = width / height
    best_match = None
    best_diff = float('inf')
    for w, h, label in STANDARD_RATIOS:
        diff = abs(ratio - w / h)
        if diff < best_diff:
            best_diff = diff
            best_match = label
    return best_match


def is_9_16(w, h, tolerance=0.02):
    """Check if dimensions are already 9:16 (within tolerance)."""
    target = TARGET_RATIO_W / TARGET_RATIO_H  # 0.5625
    return abs(w / h - target) / target < tolerance


def fit_to_canvas(img, canvas_w, canvas_h, label="image"):
    """Scale-to-fill + center-crop an image to exact canvas dimensions.

    If the image is already 9:16 it is simply resized to the canvas size
    (no cropping needed). Otherwise it is scaled up/down so the smallest
    side covers the canvas, then center-cropped to the exact size.
    """
    h, w = img.shape[:2]

    if is_9_16(w, h):
        # Already 9:16 — just resize to canvas, no crop needed
        resized = cv2.resize(img, (canvas_w, canvas_h),
                             interpolation=cv2.INTER_AREA if w > canvas_w else cv2.INTER_LANCZOS4)
        print(f"  {label}: already 9:16 ({w}x{h}), resized to {canvas_w}x{canvas_h}")
        return resized

    detected = detect_aspect_ratio(w, h)
    # Scale-to-fill: pick the larger scale so image covers entire canvas
    scale = max(canvas_w / w, canvas_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)

    # Center-crop to exact canvas size
    x_start = (new_w - canvas_w) // 2
    y_start = (new_h - canvas_h) // 2
    cropped = resized[y_start:y_start + canvas_h, x_start:x_start + canvas_w]

    print(f"  {label}: was {detected} ({w}x{h}), scaled to {new_w}x{new_h}, center-cropped to {canvas_w}x{canvas_h}")
    return cropped


def compute_canvas_size(bg_img, gs_img):
    """Compute the 9:16 canvas size.

    Uses the largest height between the two images so neither is upscaled
    more than necessary, then derives width from the 9:16 ratio.
    """
    bg_h = bg_img.shape[0]
    gs_h = gs_img.shape[0]
    canvas_h = max(bg_h, gs_h)
    canvas_w = int(canvas_h * TARGET_RATIO_W / TARGET_RATIO_H)
    # Ensure even dimensions for codec compatibility
    canvas_w = canvas_w if canvas_w % 2 == 0 else canvas_w + 1
    canvas_h = canvas_h if canvas_h % 2 == 0 else canvas_h + 1
    return canvas_w, canvas_h


def process_chromakey(bg_url, greenscreen_url, output_path,
                      spill_offset=None, min_diff=None, max_diff=None):
    """Process chromakey overlay. Returns (success, error_message, aspect_ratio) tuple.

    Pipeline:
    1. Download both images
    2. Compute a 9:16 canvas from the largest height
    3. Fit bg_image to the canvas (scale-to-fill + center-crop; no crop if already 9:16)
    4. Fit greenscreen to the canvas (same logic; no crop if already 9:16)
    5. Chromakey composite: green areas → show bg, non-green areas → keep text
    """
    print(f"  Downloading bg_image: {bg_url}")
    bg_img, bg_err = download_image(bg_url)
    if bg_img is None:
        err = f"Failed to download bg_image: {bg_err}"
        print(f"  {err}")
        return False, err, None

    print(f"  Downloading greenscreen_image: {greenscreen_url}")
    gs_img, gs_err = download_image(greenscreen_url)
    if gs_img is None:
        err = f"Failed to download greenscreen_image: {gs_err}"
        print(f"  {err}")
        return False, err, None

    # Strip alpha channel if present
    if len(bg_img.shape) == 2:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
    elif bg_img.shape[2] == 4:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    if len(gs_img.shape) == 2:
        gs_img = cv2.cvtColor(gs_img, cv2.COLOR_GRAY2BGR)
    elif gs_img.shape[2] == 4:
        gs_img = cv2.cvtColor(gs_img, cv2.COLOR_BGRA2BGR)

    bg_h, bg_w = bg_img.shape[:2]
    gs_h, gs_w = gs_img.shape[:2]
    print(f"  Original dimensions: bg={bg_w}x{bg_h}, gs={gs_w}x{gs_h}")

    # Step 1: Compute 9:16 canvas
    canvas_w, canvas_h = compute_canvas_size(bg_img, gs_img)
    print(f"  9:16 canvas: {canvas_w}x{canvas_h}")

    # Step 2: Fit both images to the canvas
    bg_img = fit_to_canvas(bg_img, canvas_w, canvas_h, label="bg_image")
    gs_img = fit_to_canvas(gs_img, canvas_w, canvas_h, label="greenscreen")

    # Both images are now exactly canvas_w x canvas_h
    # Resolve per-item overrides or use defaults
    _min_diff = min_diff if min_diff is not None else MIN_DIFF
    _max_diff = max_diff if max_diff is not None else MAX_DIFF
    _spill = spill_offset if spill_offset is not None else SPILL_OFFSET
    print(f"  Parameters: min_diff={_min_diff}, max_diff={_max_diff}, spill_offset={_spill}")

    # Convert to float for processing
    gs_float = gs_img.astype(np.float32)
    bg_float = bg_img.astype(np.float32)
    b, g, r = cv2.split(gs_float)

    # Green-difference keying: measure how much green dominates
    max_rb = np.maximum(r, b)
    diff = g - max_rb

    # Calculate transparency (0 = opaque/text, 1 = transparent/green screen)
    transparency = (diff - _min_diff) / (_max_diff - _min_diff)
    transparency = np.clip(transparency, 0.0, 1.0)

    # Restrict keying to green hue range using HSV
    hsv_img = cv2.cvtColor(gs_img, cv2.COLOR_BGR2HSV)
    h_channel = hsv_img[:, :, 0].astype(np.float32)

    # Circular hue difference
    hue_diff = np.minimum(
        np.abs(h_channel - TARGET_HUE),
        180.0 - np.abs(h_channel - TARGET_HUE)
    )

    # Weight transparency by hue proximity
    hue_weight = 1.0 - np.clip((hue_diff - HUE_HARD) / HUE_SOFT, 0.0, 1.0)
    transparency = transparency * hue_weight

    # Alpha: 1 = keep foreground (text), 0 = show background
    alpha = 1.0 - transparency

    # Spill suppression: limit green channel to max(r, b) + offset
    despilled_g = np.minimum(g, max_rb + _spill)
    fg_despilled = cv2.merge([b, despilled_g, r])

    # Final composite: text over background
    alpha_3d = cv2.merge([alpha, alpha, alpha])
    result = fg_despilled * alpha_3d + bg_float * (1.0 - alpha_3d)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Detect actual aspect ratio of the output image
    out_h, out_w = result.shape[:2]
    aspect_ratio = detect_aspect_ratio(out_w, out_h)
    print(f"  Output dimensions: {out_w}x{out_h}, detected format: {aspect_ratio}")

    # Save as PNG for lossless quality
    cv2.imwrite(output_path, result)
    print(f"  Saved result to {output_path}")
    return True, None, aspect_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Chromakey overlay: replace green screen with background image, keeping text."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file.")
    parser.add_argument("--output", default="results.json", help="Path to output JSON file.")
    args = parser.parse_args()

    supabase_token = os.environ.get("SUPABASE_TOKEN")
    if not supabase_token:
        print("Error: SUPABASE_TOKEN environment variable is not set.")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        file_content = f.read().strip()
        print(f"Raw file content: {file_content[:100]}...")
        try:
            items = json.loads(file_content)

            # Unpack nested JSON (from n8n GitHub node)
            if isinstance(items, dict) and 'images_json' in items:
                items = items['images_json']

            # GitHub Actions inputs might arrive double-stringified
            if isinstance(items, str):
                items = json.loads(items)

            if not isinstance(items, list):
                print(f"Error: Expected a list of objects, got {type(items)}")
                sys.exit(1)

        except Exception as e:
            print(f"Error parsing JSON input: {e}")
            sys.exit(1)

    results = []
    errors = []

    for idx, item in enumerate(items):
        print(f"\n--- Processing pair {idx+1}/{len(items)} ---")
        bg_url = item.get("bg_image")
        gs_url = item.get("greenscreen_image")

        if not bg_url or not gs_url:
            err_msg = "Missing bg_image or greenscreen_image in input"
            print(f"  Skipping: {err_msg}")
            errors.append({
                "index": idx,
                "bg_image": bg_url or "(missing)",
                "greenscreen_image": gs_url or "(missing)",
                "error": err_msg
            })
            continue

        # Optional per-item parameter overrides
        spill_offset = item.get("spill_offset")
        if spill_offset is not None:
            spill_offset = float(spill_offset)
        min_diff = item.get("min_diff")
        if min_diff is not None:
            min_diff = float(min_diff)
        max_diff = item.get("max_diff")
        if max_diff is not None:
            max_diff = float(max_diff)

        temp_filepath = f"temp_{uuid.uuid4().hex[:8]}.png"

        success, err_msg, detected_format = process_chromakey(bg_url, gs_url, temp_filepath,
                                    spill_offset=spill_offset,
                                    min_diff=min_diff, max_diff=max_diff)

        if success:
            filename = f"chromakey_{uuid.uuid4().hex}.png"
            print(f"  Uploading to Supabase as {filename}...")
            public_url = upload_to_supabase(temp_filepath, supabase_token, filename)

            if public_url:
                item["final_image"] = public_url
                item["detected_format"] = detected_format
                results.append(item)
            else:
                upload_err = f"Failed to upload to Supabase (file: {filename})"
                print(f"  {upload_err}")
                errors.append({
                    "index": idx,
                    "bg_image": bg_url,
                    "greenscreen_image": gs_url,
                    "error": upload_err
                })

            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        else:
            print(f"  Processing failed: {err_msg}")
            errors.append({
                "index": idx,
                "bg_image": bg_url,
                "greenscreen_image": gs_url,
                "error": err_msg
            })

    total = len(items)
    ok = len(results)
    fail = len(errors)
    summary = f"{ok}/{total} succeeded, {fail} failed"

    output_data = {
        "results": results,
        "errors": errors,
        "summary": summary
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{summary}. Output saved to {args.output}")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  [{e['index']}] {e['error']}")


if __name__ == '__main__':
    main()
