"""Nemotron Parse: load the model once, then process every page in a PDF."""
import argparse
import os
import sys
import time
import io

import fitz
import torch
from PIL import Image

from pdf_to_md import raw_to_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-path", default="/home/cqphuc/NVIDIA-Nemotron-Parse-2.0")
    ap.add_argument("--zoom", type=float, default=3.0)
    ap.add_argument("--max-new-tokens", type=int, default=4000)
    args = ap.parse_args()

    started = time.perf_counter()
    doc = fitz.open(args.pdf)
    print(f"pages={len(doc)}")

    print("loading_model=1", flush=True)
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()
    load_elapsed = time.perf_counter() - started
    print(f"model_load_elapsed={load_elapsed:.2f}s", flush=True)

    prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
    results = []
    timings = []
    for page_no in range(len(doc)):
        page_started = time.perf_counter()
        page = doc.load_page(page_no)
        pix = page.get_pixmap(matrix=fitz.Matrix(args.zoom, args.zoom), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, repetition_penalty=1.1,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = processor.decode(generated, skip_special_tokens=False)
        results.append(raw_to_markdown(raw))
        elapsed = time.perf_counter() - page_started
        timings.append(elapsed)
        print(f"page={page_no + 1}/{len(doc)} elapsed={elapsed:.2f}s", flush=True)

    with open(args.output, "w", encoding="utf-8") as out:
        out.write(f"# {os.path.basename(args.pdf)}\n\n")
        out.write("\n\n---\n\n".join(results))
        out.write("\n")

    total = time.perf_counter() - started
    print(f"output={args.output}")
    print(f"model_load_elapsed={load_elapsed:.2f}s")
    print(f"inference_total={sum(timings):.2f}s")
    print(f"inference_avg={sum(timings) / len(timings):.2f}s/page")
    print(f"total_elapsed={total:.2f}s")


if __name__ == "__main__":
    main()
