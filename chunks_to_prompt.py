"""Turn Nemotron Structured Chunks JSON into an LLM-ready prompt."""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("json_file")
ap.add_argument("--question", default="Tóm tắt các luận điểm chính và trích dẫn số liệu quan trọng.")
args = ap.parse_args()

data = json.load(open(args.json_file, encoding="utf-8"))
print("Bạn là trợ lý phân tích tài liệu. Chỉ sử dụng thông tin trong các chunks dưới đây.")
print("Nếu thiếu dữ liệu, hãy nói rõ là không tìm thấy trong tài liệu.")
print("Luôn ghi số trang và loại chunk khi trích dẫn.\n")
print("CÂU HỎI:")
print(args.question + "\n")
print("STRUCTURED CHUNKS:")
for chunk in data["chunks"]:
    print(f"\n[Trang {chunk['page'] + 1} | {chunk['type']} | order={chunk['order']} | bbox={chunk['bbox_normalized']}]")
    print(chunk["text"])
