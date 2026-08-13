"""
Test Nemotron Parse 2.0 - Xuất kết quả ra Markdown
"""
import argparse
import re
import sys
import os
import torch
import fitz  # PyMuPDF
from PIL import Image
import io


def pdf_page_to_pil_image(pdf_path, page_number=0, zoom=3.0):
    doc = fitz.open(pdf_path)
    if page_number >= len(doc):
        print(f"Lỗi: Trang {page_number} không tồn tại. File có {len(doc)} trang.")
        return None
    page = doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


# Regex to extract bbox + class segments from model output
_RE_SEGMENT = re.compile(
    r'<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>',
    re.DOTALL
)


def raw_to_markdown(raw_text: str, table_format: str = "markdown") -> str:
    """
    Convert Nemotron Parse raw output to clean Markdown.
    Strips bbox tags, converts LaTeX tables, and formats headings.
    """
    # Import model's own postprocessing utilities
    model_dir = "/home/cqphuc/NVIDIA-Nemotron-Parse-2.0"
    sys.path.insert(0, model_dir)
    from postprocessing import postprocess_text

    lines = []
    for m in _RE_SEGMENT.finditer(raw_text):
        x1, y1, text, x2, y2, cls = m.groups()
        text = text.strip()
        if not text:
            continue

        processed = postprocess_text(text, cls=cls, text_format="markdown", table_format=table_format)
        processed = processed.strip()
        if not processed:
            continue

        # Apply Markdown heading levels based on class
        if cls == "Page-header":
            lines.append(f"---\n*[Page Header: {processed}]*\n")
        elif cls == "Section-header":
            # Detect heading depth by leading #
            if processed.startswith("#"):
                lines.append(processed + "\n")
            else:
                lines.append(f"## {processed}\n")
        elif cls == "Title":
            lines.append(f"# {processed}\n")
        elif cls == "Caption":
            lines.append(f"*{processed}*\n")
        elif cls == "Table":
            lines.append(processed + "\n")
        elif cls == "Picture":
            lines.append(f"[Image]\n")
        elif cls in ("List-item",):
            if not processed.startswith("-") and not processed.startswith("*"):
                lines.append(f"- {processed}\n")
            else:
                lines.append(processed + "\n")
        else:  # Text, Footnote, Formula, etc.
            lines.append(processed + "\n")

    result = "\n".join(lines)

    # Cleanup any remaining raw tags that weren't matched by the segment regex
    result = re.sub(r'<x_[\d.]+><y_[\d.]+>', '', result)
    result = re.sub(r'<class_[^>]+>', '', result)
    result = re.sub(r'</?s>', '', result)
    result = re.sub(r'\n{3,}', '\n\n', result)  # collapse multiple blank lines

    return result.strip()


def main():
    parser = argparse.ArgumentParser(description="Nemotron Parse 2.0 → Markdown")
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--model-path", type=str, default="/home/cqphuc/NVIDIA-Nemotron-Parse-2.0")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu file .md/.txt")
    parser.add_argument("--table-format", type=str, default="markdown",
                        choices=["markdown", "latex", "HTML", "json", "csv"],
                        help="Định dạng bảng biểu (mặc định: markdown)")
    args = parser.parse_args()

    print(f"[1/4] Đang đọc trang {args.page} của: {args.pdf}")
    image = pdf_page_to_pil_image(args.pdf, args.page)
    if image is None:
        sys.exit(1)
    print(f"       Kích thước ảnh: {image.size}")

    print("[2/4] Đang nạp model và processor...")
    from transformers import AutoModel, AutoProcessor
    import warnings
    warnings.filterwarnings("ignore")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    print("       Model đã sẵn sàng!")

    print("[3/4] Đang chạy inference...")
    prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4000,
            do_sample=False,
            repetition_penalty=1.1,
        )

    print("[4/4] Đang chuyển đổi sang Markdown...")
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = processor.decode(generated, skip_special_tokens=False)

    md_content = raw_to_markdown(raw, table_format=args.table_format)

    print("\n" + "="*60)
    print("🚀 KẾT QUẢ MARKDOWN:")
    print("="*60 + "\n")
    print(md_content)
    print("\n" + "="*60)

    # Auto-determine output path
    out_path = args.output
    if out_path is None:
        pdf_stem = os.path.splitext(os.path.basename(args.pdf))[0]
        out_path = f"{pdf_stem}_page{args.page}.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n✅ Đã lưu Markdown vào: {out_path}")


if __name__ == "__main__":
    main()
