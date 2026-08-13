import requests
import json
import base64
import fitz

pdf_path = "/home/cqphuc/HVN_Baocaotaichinh_Q2_2026_Congtyme.pdf"
doc = fitz.open(pdf_path)
page = doc.load_page(0)
pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
image_bytes = pix.tobytes("png")
base64_encoded = base64.b64encode(image_bytes).decode('utf-8')

prompt = "<image><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"

payload = {
    "model": "/home/cqphuc/NVIDIA-Nemotron-Parse-2.0",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_encoded}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 1000,
    "temperature": 0.0
}

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json=payload
)

print(response.status_code)
print(response.json())
