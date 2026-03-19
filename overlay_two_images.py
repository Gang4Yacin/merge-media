import cv2
import numpy as np
import requests
import argparse
import sys
import json
import os
import uuid

SUPABASE_URL = "https://bksiaeiqzmoaxvkdtspn.supabase.co/storage/v1/object/n8n-image-generation/images"
SUPABASE_PUBLIC_URL_PREFIX = "https://bksiaeiqzmoaxvkdtspn.supabase.co/storage/v1/object/public/n8n-image-generation/images"

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    else:
        print(f"Failed to download image from {url}")
        return None

def upload_to_supabase(filepath, token, filename):
    url = f"{SUPABASE_URL}/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "image/jpeg"
    }
    with open(filepath, "rb") as f:
        response = requests.post(url, headers=headers, data=f)
    
    if response.status_code in [200, 201]:
        return f"{SUPABASE_PUBLIC_URL_PREFIX}/{filename}"
    else:
        print(f"Failed to upload {filename} to Supabase: {response.text}")
        return None

def process_single_image(fg_url, ad_url, output_path):
    print(f"  Downloading fg_image: {fg_url}")
    overlay_img = download_image(fg_url)
    if overlay_img is None: return False
    
    print(f"  Downloading ad_image: {ad_url}")
    ad_img = download_image(ad_url)
    if ad_img is None: return False
    
    if ad_img.shape[2] == 4:
        ad_img = cv2.cvtColor(ad_img, cv2.COLOR_BGRA2BGR)
    
    if overlay_img.shape[2] != 4:
        print("  Error: foreground image has no alpha channel.")
        return False
        
    alpha_channel = overlay_img[:, :, 3]
    transparent_mask = (alpha_channel < 255).astype(np.uint8) * 255
    contours, _ = cv2.findContours(transparent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("  Error: No transparent area found.")
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    ad_aspect = ad_img.shape[1] / ad_img.shape[0]
    box_aspect = w / h
    
    if ad_aspect > box_aspect:
        new_h = h
        new_w = int(h * ad_aspect)
        resized = cv2.resize(ad_img, (new_w, new_h))
        crop_x = (new_w - w) // 2
        ad_resized = resized[:, crop_x:crop_x+w]
    else:
        new_w = w
        new_h = int(w / ad_aspect)
        resized = cv2.resize(ad_img, (new_w, new_h))
        crop_y = (new_h - h) // 2
        ad_resized = resized[crop_y:crop_y+h, :]

    ad_resized = cv2.resize(ad_resized, (w, h))

    canvas = np.zeros((overlay_img.shape[0], overlay_img.shape[1], 3), dtype=np.uint8)
    canvas[y:y+h, x:x+w] = ad_resized

    alpha = overlay_img[:, :, 3] / 255.0
    alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
    
    fg = overlay_img[:, :, :3]
    bg = canvas
    
    result = (fg * alpha_3d + bg * (1 - alpha_3d)).astype(np.uint8)
    
    cv2.imwrite(output_path, result)
    return True

def main():
    parser = argparse.ArgumentParser(description="Overlay multiple ad images behind valid transparent PNGs.")
    parser.add_argument("--input", required=True, help="Path to input JSON file containing array of image pairs.")
    parser.add_argument("--output", default="results.json", help="Path to output JSON file.")
    args = parser.parse_args()

    supabase_token = os.environ.get("SUPABASE_TOKEN")
    if not supabase_token:
        print("Error: SUPABASE_TOKEN environment variable is not set.")
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        file_content = f.read().strip()
        print(f"Raw file content: {file_content[:100]}...") # Print beginning of JSON for debugging
        try:
            items = json.loads(file_content)
            
            # Unpack JSON if it's nested
            if isinstance(items, dict) and 'images_json' in items:
                items = items['images_json']
                
            # GitHub actions inputs might arrive stringified twice
            if isinstance(items, str):
                items = json.loads(items)
                
            if not isinstance(items, list):
                print(f"Error: Expected a list of objects, got {type(items)}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error parsing JSON input: {e}")
            sys.exit(1)

    results = []
    
    for idx, item in enumerate(items):
        print(f"\n--- Processing pair {idx+1}/{len(items)} ---")
        fg_url = item.get("fg_image")
        ad_url = item.get("ad_image")
        
        if not fg_url or not ad_url:
            print("  Skipping: missing fg_image or ad_image")
            continue
            
        temp_filepath = f"temp_{uuid.uuid4().hex[:8]}.jpg"
        
        success = process_single_image(fg_url, ad_url, temp_filepath)
        
        if success:
            filename = f"overlay_{uuid.uuid4().hex}.jpg"
            print(f"  Uploading to Supabase as {filename}...")
            public_url = upload_to_supabase(temp_filepath, supabase_token, filename)
            
            if public_url:
                item["final_image"] = public_url
                if "meta_ad_creative_id" not in item:
                    item["meta_ad_creative_id"] = None
                results.append(item)
            else:
                print("  Upload failed, skipping from results.")
            
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        else:
            print("  Processing failed, skipping from results.")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted processing. {len(results)} successful overlays saved to {args.output}")

if __name__ == '__main__':
    main()
