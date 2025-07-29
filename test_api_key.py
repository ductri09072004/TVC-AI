from openai import OpenAI

# Thay thế bằng API key thật của bạn
api_key = input("Nhập API key của bạn: ").strip()

try:
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    
    print("✅ API key hợp lệ!")
    print("Response:", response.choices[0].message.content)
    
except Exception as e:
    print("❌ API key không hợp lệ hoặc có lỗi:")
    print(e) 