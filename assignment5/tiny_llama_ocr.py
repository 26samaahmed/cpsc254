import zipfile
import os
import pytesseract
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


zip_path = "L05_DL_Vision_Receipts.zip"
extract_dir = "./receipts"
model_path = "./results/checkpoint-10"
hf_token = "token here"


if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)


tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_path, token=hf_token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


image_files = sorted([f for f in os.listdir(extract_dir) if f.lower().endswith((".jpg", ".png"))])

for filename in image_files:
    image_path = os.path.join(extract_dir, filename)
    print(f"\n📄 Processing: {filename}")
    
    # OCR
    ocr_text = pytesseract.image_to_string(Image.open(image_path))

    # Prompt
    prompt = f"""### OCR Text:
{ocr_text}

### Question: List all purchased items with their prices and calculate the total cost.
### Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = output_text.replace(prompt, "").strip()

    # Print results
    print("\n🧾 OCR Text:")
    print(ocr_text.strip())
    print("\n🤖 Extracted Info:")
    print(answer)
    print("=" * 60)
