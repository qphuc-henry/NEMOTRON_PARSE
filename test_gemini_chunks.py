"""Send Nemotron Structured Chunks to Gemini 2.5 Flash via the REST API."""
import argparse
import json
import os
import sys

import requests


ap = argparse.ArgumentParser()
ap.add_argument("--chunks", required=True)
ap.add_argument("--question", required=True)
ap.add_argument("--model", default="gemini-2.5-flash")
ap.add_argument("--output", required=True)
args = ap.parse_args()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Thiếu GEMINI_API_KEY. Hãy export GEMINI_API_KEY='...' trước khi chạy.")

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

url = f"https://generativelanguage.googleapis.com/v1beta/models/{args.model}:generateContent?key={api_key}"
payload = {"contents": [{"parts": [{"text": "\n".join(parts)}]}]}
response = requests.post(url, json=payload, timeout=120)
response.raise_for_status()
result = response.json()
answer = result["candidates"][0]["content"]["parts"][0]["text"]
open(args.output, "w", encoding="utf-8").write(answer + "\n")
print(f"output={args.output}")
print(answer)
