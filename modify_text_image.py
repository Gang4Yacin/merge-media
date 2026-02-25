import os
import requests
from PIL import Image
from io import BytesIO
import time
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai
from google.genai.errors import APIError

PROMPT = """Tu recevras en entrée l'url d'une image contenant une partie avec plein de "Text1 Text1 Text1 Text1" écris.
Tu devras remplacer cette partie par "Mais c'est excellent, j'adore ça !!" en gardant exactement la même police d'écriture, la même couleur et les mêmes espaces.
Tu modifieras seulement cette partie et gardera l'image exactement comme tu l'as reçu. Si tu dois ajouter des emojis, tu ajouteras les emojis au format iOS."""

def get_image(url):
    print(f"Downloading image from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate_modification(client, model_name, img, prompt):
    print(f"Calling Google GenAI API with model {model_name}...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[img, prompt]
        )
        return response
    except Exception as e:
        print(f"API Error occurred: {e}. Retrying if possible...")
        raise

def main():
    parser = argparse.ArgumentParser(description="Modify text in image using Gemini API.")
    parser.add_argument("--image_url", required=True, help="URL of the image to modify.")
    parser.add_argument("--output", default="output_modified.jpg", help="Output image path.")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        exit(1)

    try:
        img = get_image(args.image_url)
        client = genai.Client(api_key=api_key)
        model_name = "gemini-3-pro-image-preview"
        
        response = generate_modification(client, model_name, img, PROMPT)
        
        # Check and extract the generated image
        saved = False
        if response.candidates and response.candidates[0].content.parts:
            for i, part in enumerate(response.candidates[0].content.parts):
                if hasattr(part, 'inline_data') and part.inline_data:
                    mime_type = part.inline_data.mime_type
                    # The SDK stores bytes in data
                    img_data = part.inline_data.data
                    with open(args.output, "wb") as f:
                        f.write(img_data)
                    print(f"Successfully saved modified image to {args.output}")
                    saved = True
                    break
                    
        if not saved:
            print("No image data found in the response. The model may have returned text only.")
            print("Text response:", response.text)

    except Exception as e:
        print(f"Process failed after multiple retries: {e}")
        exit(1)

if __name__ == "__main__":
    main()
