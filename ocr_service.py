import base64
from openai import OpenAI
from utils_text import normalize_vertical_text


def extract_text_from_image_gpt41mini(client: OpenAI, image_bytes: bytes, mime: str) -> str:
    """
    gpt-4.1-mini 기반 OCR
    - data URL(image_url)로 전달 (image_base64 사용 X)
    - 세로 텍스트는 가로로 정리
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime};base64,{b64}"

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "이 이미지에서 보이는 모든 텍스트를 추출하되, "
                            "세로로 배치된 글자들은 사람이 읽기 쉬운 가로 문장으로 재구성해줘. "
                            "의미 없는 한 글자씩의 줄바꿈은 제거하고, "
                            "자연스러운 문장 단위로 공백을 사용해 표현해줘. "
                            "추가 설명 없이 결과 텍스트만 출력해."
                        )
                    },
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )

    raw = (resp.output_text or "").strip()
    return normalize_vertical_text(raw)
