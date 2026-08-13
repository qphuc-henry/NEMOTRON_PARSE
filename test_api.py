from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1" 
)

print("Gửi request tới model NVIDIA-Nemotron-Parse-2.0...")

# Gọi model để test trích xuất từ một ảnh ví dụ
try:
    response = client.chat.completions.create(
        model="/home/cqphuc/NVIDIA-Nemotron-Parse-2.0",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all structured text and tables from this document image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            # Có thể thay thế bằng link PDF/ảnh thực tế hoặc base64
                            "url": "https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png"
                        },
                    },
                ],
            }
        ],
        max_tokens=1024,
    )

    print("\n[KẾT QUẢ TRẢ VỀ]:\n")
    print(response.choices[0].message.content)
except Exception as e:
    print("\n[LỖI]: Không kết nối được. Hãy đảm bảo bạn đã chạy ./start_vllm.sh và server đã sẵn sàng!")
    print(e)
