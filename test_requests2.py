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

# The prompt MUST NOT have </s><s> as text - those need to be actual token IDs
# Pass the image FIRST, then the control tokens
payload = {
    "model": "/home/cqphuc/NVIDIA-Nemotron-Parse-2.0",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_encoded}"
                    }
                },
                {"type": "text", "text": "<predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"},
            ]
        }
    ],
    "max_tokens": 1000,
    "temperature": 0.0,
    "extra_body": {
        "repetition_penalty": 1.1,
        "top_k": 1,
        "skip_special_tokens": False,
    }
}

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json=payload
)

result = response.json()
print("STATUS:", response.status_code)
print("PROMPT_TOKENS:", result.get("usage", {}).get("prompt_tokens"))
content = result["choices"][0]["message"]["content"]
print("LEN:", len(content))
print("CONTENT[:500]:", repr(content[:500]))
