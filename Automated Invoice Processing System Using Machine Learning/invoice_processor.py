
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def process_invoice(filepath):
    # Convert PDF to image (if PDF)
    if filepath.endswith('.pdf'):
        pages = convert_from_path(filepath)
        image = pages[0]  # Take only the first page
    else:
        image = Image.open(filepath)

    # OCR using pytesseract
    text = pytesseract.image_to_string(image)

    # Basic data extraction using regex (can be improved)
    invoice_no = re.search(r'Invoice\s*No[:\-]?\s*(\S+)', text, re.IGNORECASE)
    date = re.search(r'Date[:\-]?\s*([0-9]{2}/[0-9]{2}/[0-9]{4})', text)
    amount = re.search(r'Total\s*[:\-]?\s*(Rs\.?|₹)?\s*([0-9,]+\.\d{2})', text)

    return {
        'invoice_number': invoice_no.group(1) if invoice_no else 'Not found',
        'date': date.group(1) if date else 'Not found',
        'total_amount': amount.group(2) if amount else 'Not found',
        'full_text': text
    }


