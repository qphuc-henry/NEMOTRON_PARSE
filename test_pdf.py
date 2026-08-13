import base64
import argparse
import fitz  # PyMuPDF
from openai import OpenAI

def pdf_page_to_base64(pdf_path, page_number=0):
    """
    Đọc một trang cụ thể của file PDF và chuyển nó thành ảnh Base64
    """
    try:
        # Mở file PDF
        doc = fitz.open(pdf_path)
        
        if page_number >= len(doc):
            print(f"Lỗi: Trang {page_number} không tồn tại. File PDF này chỉ có {len(doc)} trang.")
            return None
            
        page = doc.load_page(page_number)
        
        # Render trang thành ảnh với độ phân giải cao (zoom x2 để model đọc rõ nét hơn)
        zoom = 2.0 
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert ảnh sang chuỗi bytes và encode thành base64
        image_bytes = pix.tobytes("png")
        base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
        
        return base64_encoded
        
    except Exception as e:
        print(f"Lỗi khi đọc file PDF: {e}")
        return None


def extract_content_from_image(base64_image, model_name="/home/cqphuc/NVIDIA-Nemotron-Parse-2.0"):
    """
    Gửi ảnh lên vLLM Server để trích xuất dữ liệu
    """
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1" 
    )

    print("Đang gửi yêu cầu tới model NVIDIA-Nemotron-Parse-2.0 (vLLM)...")
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=2000,
            extra_body={
                "repetition_penalty": 1.1,
                "top_k": 1,
                "skip_special_tokens": False,
            }
        )


        return response.choices[0].message.content
        
    except Exception as e:
        print("\n[LỖI]: Không kết nối được. Hãy đảm bảo bạn đã chạy ./start_vllm.sh và server đã sẵn sàng!")
        print(e)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG Pipeline với file PDF")
    parser.add_argument("--pdf", type=str, required=True, help="Đường dẫn tới file PDF cần test")
    parser.add_argument("--page", type=int, default=0, help="Trang số mấy trong PDF (bắt đầu từ 0)")
    
    args = parser.parse_args()
    
    print(f"Đang xử lý trang {args.page} của file PDF: {args.pdf}")
    
    # Bước 1: Chuyển đổi PDF -> Base64 Image
    base64_img = pdf_page_to_base64(args.pdf, args.page)
    
    # Bước 2: Gửi Base64 Image -> Model để lấy Text/Markdown
    if base64_img:
        result = extract_content_from_image(base64_img)
        
        print("\n" + "="*50)
        print("🚀 KẾT QUẢ TRÍCH XUẤT (MARKDOWN / STRUCTURED DATA):")
        print("="*50 + "\n")
        print(f"RAW RESULT: {repr(result)}")
        print(result)
        print("\n" + "="*50)
