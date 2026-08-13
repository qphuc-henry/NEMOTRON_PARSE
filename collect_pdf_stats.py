import glob, os, re, statistics

ROOT = os.path.dirname(os.path.abspath(__file__))

docs = [
    ("48-bxd.signed", 26),
    ("BCPT", 14),
    ("HVN_Baocaotaichinh_Q2_2026_Congtyme", 30),
]

def classify(text):
    has_table = bool(re.search(r'^\s*\|.*\|\s*$', text, re.M))
    has_heading = bool(re.search(r'^\s*#{1,6}\s+', text, re.M))
    has_bullets = bool(re.search(r'^\s*[-*]\s+', text, re.M))
    if has_table and (has_heading or has_bullets):
        return "hỗn hợp (văn bản + bảng)"
    if has_table:
        return "bảng"
    return "văn bản"

all_rows = []
for stem, pages in docs:
    timing_path = os.path.join(ROOT, f"{stem}.timing.tsv")
    rows = []
    with open(timing_path, encoding="utf-8") as f:
        for line in f:
            page, seconds, rc = line.rstrip().split("\t")
            md = os.path.join(ROOT, f"{stem}_page{int(page)}_test.md")
            text = open(md, encoding="utf-8").read() if os.path.exists(md) else ""
            rows.append((int(page), float(seconds), classify(text), int(rc)))
    all_rows.extend((stem, *r) for r in rows)

with open(os.path.join(ROOT, "pdf_timing_stats.md"), "w", encoding="utf-8") as out:
    out.write("# Thống kê thời gian Nemotron Parse\n\n")
    out.write("Thời gian tính từ lúc gọi `pdf_to_md.py` cho từng trang đến khi lưu xong Markdown.\n\n")
    out.write("## Theo tài liệu\n\n| Tài liệu | Trang | Tổng (giây) | Trung bình/trang (giây) | Nhanh nhất | Chậm nhất |\n|---|---:|---:|---:|---:|---:|\n")
    for stem, pages in docs:
        vals = [r[1] for s, *r in all_rows if s == stem]
        out.write(f"| `{stem}.pdf` | {len(vals)} | {sum(vals):.2f} | {statistics.mean(vals):.2f} | {min(vals):.2f} | {max(vals):.2f} |\n")
    vals = [r[1] for _, *r in all_rows]
    out.write(f"| **Tổng cả 3** | **{len(vals)}** | **{sum(vals):.2f}** | **{statistics.mean(vals):.2f}** | — | — |\n\n")
    out.write("## Theo loại trang\n\n| Tài liệu | Loại trang | Số trang | Tổng (giây) | Trung bình/trang (giây) |\n|---|---|---:|---:|---:|\n")
    for stem, pages in docs:
        kinds = sorted(set(r[2] for s, *r in all_rows if s == stem))
        for kind in kinds:
            vals = [r[1] for s, *r in all_rows if s == stem and r[2] == kind]
            out.write(f"| `{stem}.pdf` | {kind} | {len(vals)} | {sum(vals):.2f} | {statistics.mean(vals):.2f} |\n")
    out.write("\n## Dữ liệu chi tiết\n\n| Tài liệu | Trang (0-based) | Loại | Thời gian (giây) | Mã lỗi |\n|---|---:|---|---:|---:|\n")
    for stem, page, seconds, kind, rc in all_rows:
        out.write(f"| `{stem}.pdf` | {page} | {kind} | {seconds:.2f} | {rc} |\n")

print("wrote " + os.path.join(ROOT, "pdf_timing_stats.md"))
