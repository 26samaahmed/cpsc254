import os
import zipfile
import pytesseract
from PIL import ImageOps, Image
import pandas as pd
import re
from collections import defaultdict

# Extract ZIP
zip_path = 'L05_DL_Vision_Receipts.zip'
extract_dir = 'receipts'

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Output data
data = []
store_totals = defaultdict(float)

# Keywords to ignore
ignore_keywords = ['subtotal', 'total', 'tend', 'change', 'feedback', 'debit', 'cash']

def extract_info_from_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    store_name = "Unknown Store"

    # Store name from any of the top 10 lines
    for line in lines[:10]:
        l = line.lower()
        if "walmart" in l:
            store_name = "WALMART"
            break
        elif "target" in l:
            store_name = "TARGET"
            break
        elif re.search(r'\b(store|mart|market)\b', l):
            store_name = re.sub(r'[^A-Za-z ]', '', line.strip()).upper()
            break

    item_pattern = re.compile(r'^(.+?)\s+(\d+\.\d{2})[^\d]*$')


    for line in lines:
        line_clean = line.strip()
        lower_line = line_clean.lower()

        # Skip common non-item lines
        if any(word in lower_line for word in ['subtotal', 'total', 'tax', 'change', 'tend', 'feedback', '@', '%', 'discount', 'debit']):
            continue

        # Skip suspiciously short lines or weird symbols
        if len(line_clean) < 5 or re.search(r'[^A-Za-z0-9\s\.\:]', line_clean):
            continue


        match = item_pattern.match(line_clean)
        if match:
            item = re.sub(r'\d{6,}', '', match.group(1)).strip().rstrip('.:')
            item = re.sub(r'\s+F$', '', item)  # Remove any trailing ' F' from items because there were no items with 'F' in the dataset
            item = re.sub(r'[@\d]+', '', item).strip()  # Remove unwanted quantities or symbols after the item name
            item = re.sub(r'(\b\w+)\s+(\d+)\s+(\w+)', r'\1 \2 \3', item)
            item = re.sub(r'(\w)\s*x\s*(\d+)', r'\1x\2', item)
            item = re.sub(r'OT\s+(\d+)\s+(\w+)', r'OT \1 \2', item)

            try:
                amount = float(match.group(2))
                data.append((store_name, item, amount))
                store_totals[store_name] += amount
            except ValueError:
                continue


# Run OCR on each image
for filename in os.listdir(extract_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(extract_dir, filename)
        try:
            image = Image.open(img_path).convert('L')
            image = ImageOps.invert(image)  # Invert for better OCR on light backgrounds
            text = pytesseract.image_to_string(image)


            extract_info_from_text(text)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

df = pd.DataFrame(data, columns=['Store', 'Item', 'Amount'])

for store, total in store_totals.items():
    df = pd.concat([df, pd.DataFrame([{
        'Store': store,
        'Item': 'TOTAL',
        'Amount': round(total, 2)
    }])], ignore_index=True)


df.to_csv('shopping_summary.csv', index=False)
print(df)
print("shopping_summary.csv created with filtered item list.")
