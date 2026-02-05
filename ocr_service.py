import google.generativeai as genai
from utils_text import normalize_vertical_text

def extract_text_from_image_gemini(image_bytes: bytes, mime: str) -> str:
    """
    Gemini 1.5 Flash 기반 OCR
    - 별도의 base64 string 변환 없이 bytes 데이터를 직접 사용
    - 세로 텍스트 가로 정리 및 자연스러운 문장 재구성
    """
    
    # 1. 모델 설정 (OCR에는 속도가 빠른 flash 모델을 추천합니다)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # 2. 프롬프트 구성
    prompt = (
        "이 이미지에서 보이는 모든 텍스트를 추출하되, "
        "세로로 배치된 글자들은 사람이 읽기 쉬운 가로 문장으로 재구성해줘. "
        "의미 없는 한 글자씩의 줄바꿈은 제거하고, "
        "자연스러운 문장 단위로 공백을 사용해 표현해줘. "
        "추가 설명 없이 결과 텍스트만 출력해."
    )

    # 3. 이미지 데이터 구성
    image_part = {
        "mime_type": mime,
        "data": image_bytes
    }

    # 4. 콘텐츠 생성 요청
    response = model.generate_content([prompt, image_part])
    
    raw = (response.text or "").strip()
    
    # 기존에 사용하시던 세로 텍스트 정규화 함수 적용
    return normalize_vertical_text(raw)