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


def process_chromakey(bg_url, greenscreen_url, output_path,
                      spill_offset=None, min_diff=None, max_diff=None):
    """Process chromakey overlay. Returns (success, error_message) tuple."""
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

    # Center-align greenscreen on background canvas (no stretch/compress)
    bg_h, bg_w = bg_img.shape[:2]
    gs_h, gs_w = gs_img.shape[:2]
    print(f"  Dimensions: bg={bg_w}x{bg_h}, gs={gs_w}x{gs_h}")

    # Create a canvas the size of bg, filled with pure green (#00b140 → BGR: 64,177,0)
    # so that areas outside the greenscreen are keyed out as transparent (#00FF00 → BGR: 0,255,0)
    gs_canvas = np.full((bg_h, bg_w, 3), (0, 255, 0), dtype=np.uint8)

    # Calculate offsets to center the greenscreen on the canvas
    x_offset = (bg_w - gs_w) // 2  # negative if gs is wider than bg
    y_offset = (bg_h - gs_h) // 2  # negative if gs is taller than bg

    # Source region (crop from gs if it overflows the canvas)
    src_x1 = max(0, -x_offset)
    src_y1 = max(0, -y_offset)
    src_x2 = min(gs_w, bg_w - x_offset)
    src_y2 = min(gs_h, bg_h - y_offset)

    # Destination region on the canvas
    dst_x1 = max(0, x_offset)
    dst_y1 = max(0, y_offset)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    gs_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = gs_img[src_y1:src_y2, src_x1:src_x2]
    gs_img = gs_canvas

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
