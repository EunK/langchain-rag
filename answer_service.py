from typing import List, Dict, Any
from openai import OpenAI
from utils_text import robust_json_loads


def openai_answer_with_rag(
    client: OpenAI,
    model: str,
    question: str,
    contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    contexts: [{"page_number": int, "content": str, "similarity": float}, ...]
    return: {"answer": str, "cited_pages": [int, ...]}
    """
    ctx_lines = []
    for c in contexts:
        ctx_lines.append(f"[page={c['page_number']}, similarity={c['similarity']:.3f}]\n{c['content']}")
    ctx_text = "\n\n---\n\n".join(ctx_lines)

    system = (
        """너는 장비 매뉴얼 PDF에서 추출된 정보만을 근거로 답변하는 기술지원 전문가다.
            아래 제공되는 "매뉴얼 발췌"는 네가 참조할 수 있는 유일한 지식 창고다.

            [역할 및 지침]
            - 각 발췌문 상단의 [page=N] 태그는 해당 정보의 원본 페이지 번호다.
            - 사용자의 질문에 대해 "매뉴얼 발췌"에 명시된 절차와 수치만을 사용하여 답변한다.

            [절대 규칙]
            1. 정보 부재 시 대응: 질문에 대한 직접적인 해결책이 발췌 내용에 없다면, 추측하지 말고 반드시 아래 JSON 구조를 반환하라:
                {"answer": "문서에 존재하지 않습니다.", "cited_pages": []}

            2. 단계별 가이드: 답변은 반드시 한국어로, 실행 가능한 단계별(1., 2., 3.) 번호 목록으로 작성하라. 사족이나 배경 설명은 모두 배제한다.

            3. 정확한 인용: 답변에 사용된 정보가 포함된 모든 [page=N]의 N 숫자를 `cited_pages` 배열에 수집하라. 중복된 번호는 한 번만 기록한다.
            4. 형식 엄수: 오직 순수한 JSON 객체 하나만 출력하라. 마크다운 코드 블록(```json)이나 앞뒤 설명을 절대 포함하지 마라. 출력의 시작은 '{', 끝은 '}'여야 한다.
            
            [검증 체크리스트]
            - 답변의 모든 문장이 발췌문에 근거하는가?
            - 외부 지식이나 상식으로 내용을 보충하지 않았는가?
            - 출력 형식이 유효한 JSON인가?"""
        )

    user = f"사용자 질문:\n{question}\n\n매뉴얼 발췌:\n{ctx_text}\n"

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text_out = resp.output_text

    data = robust_json_loads((text_out or "").strip())
    if not data or "answer" not in data:
        return {"answer": "문서에 존재하지 않습니다.", "cited_pages": []}

    raw_pages = data.get("cited_pages", [])
    cited_pages: List[int] = []
    for p in raw_pages:
        try:
            cited_pages.append(int(p))
        except Exception:
            pass

    cited_pages = sorted(set(cited_pages))
    return {"answer": str(data.get("answer", "")).strip(), "cited_pages": cited_pages}
