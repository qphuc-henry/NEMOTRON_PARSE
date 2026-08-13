"""Send Nemotron Structured Chunks to an OpenAI-compatible custom LLM endpoint."""
import argparse
import json
import os
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", required=True)
ap.add_argument("--question", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--model", default="gemini/gemini-2.5-flash")
ap.add_argument("--base-url", default=None)
args = ap.parse_args()

key = os.environ.get("CUSTOM_LLM_API_KEY")
base_url = (args.base_url or os.environ.get("CUSTOM_LLM_BASE_URL", "")).rstrip("/")
if not key:
    raise SystemExit("Thiếu CUSTOM_LLM_API_KEY")
if not base_url:
    raise SystemExit("Thiếu CUSTOM_LLM_BASE_URL")

data = json.load(open(args.chunks, encoding="utf-8"))
parts = [
    "Bạn là trợ lý phân tích tài liệu. Chỉ sử dụng thông tin trong Structured Chunks.",
    "Nếu không đủ dữ liệu, hãy nói rõ. Khi trả lời, phải dẫn số trang và loại chunk.",
    "\nCÂU HỎI:\n" + args.question,
    "\nSTRUCTURED CHUNKS:",
]
for c in data["chunks"]:
    parts.append(
        f"\n[Trang {c['page'] + 1} | {c['type']} | order={c['order']} | bbox={c['bbox_normalized']}]\n"
        + c["text"]
    )

payload = {
    "model": args.model,
    "messages": [{"role": "user", "content": "\n".join(parts)}],
    "temperature": 0.0,
}
response = requests.post(
    base_url + "/chat/completions",
    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    json=payload,
    timeout=180,
)
response.raise_for_status()
result = response.json()
answer = result["choices"][0]["message"]["content"]
open(args.output, "w", encoding="utf-8").write(answer + "\n")
print(f"output={args.output}")
print(answer)
