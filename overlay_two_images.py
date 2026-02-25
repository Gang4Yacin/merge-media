import cv2
import numpy as np
import requests
import argparse
import sys

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        # IMREAD_UNCHANGED ensures alpha channel is kept if it exists
        return cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
    else:
        print(f"Failed to download image from {url}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Overlay an ad image behind a transparent PNG.")
    parser.add_argument("--fg_image", required=True, help="URL of the transparent foreground image (PNG).")
    parser.add_argument("--ad_image", required=True, help="URL of the ad image to put in the background.")
    parser.add_argument("--output", default="output_overlay.jpg", help="Output filename.")
    
    args = parser.parse_args()

    print(f"Downloading foreground image from: {args.fg_image}")
    overlay_img = download_image(args.fg_image)
    
    print(f"Downloading ad image from: {args.ad_image}")
    ad_img = download_image(args.ad_image)
    
    # If the ad image has 4 channels, convert it to 3 channels to safely resize and use it as a background.
    if ad_img.shape[2] == 4:
        ad_img = cv2.cvtColor(ad_img, cv2.COLOR_BGRA2BGR)
    
    if overlay_img.shape[2] != 4:
        print("Error: The foreground image does not have an alpha channel (transparency).")
        sys.exit(1)
        
    print("Processing images...")
    
    alpha_channel = overlay_img[:, :, 3]
    # Find transparent pixels to determine bounding box
    transparent_mask = (alpha_channel < 255).astype(np.uint8) * 255
    contours, _ = cv2.findContours(transparent_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No transparent area found in the foreground image.")
        sys.exit(1)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Resize and crop ad to transparent area flawlessly
    ad_aspect = ad_img.shape[1] / ad_img.shape[0] # width / height
    box_aspect = w / h
    
    if ad_aspect > box_aspect:
        # Ad is wider, scale height to fit, crop width
        new_h = h
        new_w = int(h * ad_aspect)
        resized = cv2.resize(ad_img, (new_w, new_h))
        crop_x = (new_w - w) // 2
        ad_resized = resized[:, crop_x:crop_x+w]
    else:
        # Ad is taller, scale width to fit, crop height
        new_w = w
        new_h = int(w / ad_aspect)
        resized = cv2.resize(ad_img, (new_w, new_h))
        crop_y = (new_h - h) // 2
        ad_resized = resized[crop_y:crop_y+h, :]

    ad_resized = cv2.resize(ad_resized, (w, h))

    # Create canvas
    canvas = np.zeros((overlay_img.shape[0], overlay_img.shape[1], 3), dtype=np.uint8)
    canvas[y:y+h, x:x+w] = ad_resized

    # Overlay
    alpha = overlay_img[:, :, 3] / 255.0
    alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
    
    fg = overlay_img[:, :, :3]
    bg = canvas
    
    result = (fg * alpha_3d + bg * (1 - alpha_3d)).astype(np.uint8)
    
    cv2.imwrite(args.output, result)
    print(f"Successfully generated and saved: {args.output}")

if __name__ == '__main__':
    main()
