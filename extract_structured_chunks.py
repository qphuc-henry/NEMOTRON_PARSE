"""Run Nemotron on one PDF page and preserve structured chunks as JSON."""
import argparse
import io
import json
import re
import time

import fitz
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


END_SEGMENT = re.compile(r"<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>")
START_BBOX = re.compile(r"^\s*<x_([\d.]+)><y_([\d.]+)>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", type=int, required=True, help="0-based page index")
    ap.add_argument("--output", required=True)
    ap.add_argument("--raw-output", required=True)
    ap.add_argument("--model-path", default="/home/cqphuc/NVIDIA-Nemotron-Parse-2.0")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    page = doc.load_page(args.page)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()
    prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=4000, do_sample=False, repetition_penalty=1.1
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(generated, skip_special_tokens=False)

    chunks = []
    cursor = 0
    for order, match in enumerate(END_SEGMENT.finditer(raw)):
        body = raw[cursor:match.start()]
        start = START_BBOX.match(body)
        if start:
            x1, y1 = start.groups()
            text = body[start.end():]
        else:
            x1, y1, text = "0", "0", body
        x2, y2, cls = match.groups()
        text = re.sub(r"</?s>|<class_[^>]+>", "", text)
        chunks.append({
            "page": args.page,
            "order": order,
            "type": cls,
            "bbox_normalized": [float(x1), float(y1), float(x2), float(y2)],
            "text": text.strip(),
        })
        cursor = match.end()

    with open(args.raw_output, "w", encoding="utf-8") as f:
        f.write(raw)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "document": args.pdf,
            "page": args.page,
            "page_size_px": list(image.size),
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "chunks": chunks,
        }, f, ensure_ascii=False, indent=2)
    print(f"chunks={len(chunks)}")
    print(f"json={args.output}")
    print(f"raw={args.raw_output}")


if __name__ == "__main__":
    main()
