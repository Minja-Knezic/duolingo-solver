import pytesseract
from PIL import Image

def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract text from a PIL Image using Tesseract OCR.

    Args:
        image (PIL.Image.Image): Image to process.

    Returns:
        str: Extracted text.
    """
    text = pytesseract.image_to_string(image)
    return text.strip()
