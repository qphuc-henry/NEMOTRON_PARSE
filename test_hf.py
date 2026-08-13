"""
Test Nemotron Parse 2.0 trực tiếp qua HuggingFace Transformers
(Không cần vLLM server)
"""
import argparse
import base64
import sys
import torch
import fitz  # PyMuPDF
from PIL import Image
import io

def pdf_page_to_pil_image(pdf_path, page_number=0, zoom=2.0):
    """Chuyển trang PDF thành PIL Image"""
    doc = fitz.open(pdf_path)
    if page_number >= len(doc):
        print(f"Lỗi: Trang {page_number} không tồn tại. File có {len(doc)} trang.")
        return None
    page = doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Test Nemotron Parse 2.0 via HuggingFace")
    parser.add_argument("--pdf", type=str, required=True)
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--model-path", type=str, default="/home/cqphuc/NVIDIA-Nemotron-Parse-2.0")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn file .txt để lưu kết quả (tuỳ chọn)")
    args = parser.parse_args()

    print(f"[1/4] Đang đọc trang {args.page} của: {args.pdf}")
    image = pdf_page_to_pil_image(args.pdf, args.page)
    if image is None:
        sys.exit(1)
    print(f"       Kích thước ảnh: {image.size}")

    print("[2/4] Đang nạp model và processor (lần đầu sẽ lâu một chút)...")
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    print("       Model đã sẵn sàng!")

    print("[3/4] Đang xử lý ảnh và chạy inference...")
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

    print("[4/4] Đang giải mã kết quả...")
    # Chỉ decode phần output mới (bỏ phần prompt)
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    result = processor.decode(generated, skip_special_tokens=False)

    print("\n" + "="*60)
    print("🚀 KẾT QUẢ TRÍCH XUẤT (MARKDOWN / STRUCTURED DATA):")
    print("="*60 + "\n")
    print(result)
    print("\n" + "="*60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n✅ Đã lưu kết quả vào: {args.output}")


if __name__ == "__main__":
    main()
