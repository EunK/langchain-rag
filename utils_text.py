import json
import re
from typing import List, Optional, Dict, Any


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def robust_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def is_refusal_answer(answer: str) -> bool:
    """
    문서 범위 밖 답변(거절) 판정 강화
    - 표현이 조금 달라도 잡히도록 정규식 포함
    """
    if not answer:
        return True

    a = (answer or "").strip()

    if "문서에 존재하지 않습니다" in a:
        return True

    patterns = [
        r"존재하지\s*않습니다",
        r"찾을\s*수\s*없습니다",
        r"확인할\s*수\s*없습니다",
        r"근거(가|를)\s*찾을\s*수\s*없습니다",
        r"(문서|매뉴얼).{0,20}없습니다",
        r"(문서|매뉴얼).{0,20}존재하지",
    ]
    return any(re.search(p, a) for p in patterns)


def merge_pages_cited_then_search(
    cited_pages: List[int],
    contexts: List[Dict[str, Any]],
    max_pages: int,
    *,
    top1_similarity: Optional[float] = None,
    min_abs: float = 0.35,
    max_drop: float = 0.08,
) -> List[int]:
    """
    1) cited_pages 우선 (무조건 포함)
    2) 부족하면 contexts로 보충하되,
       - 절대 기준: similarity >= min_abs
       - 상대 기준: similarity >= (top1_similarity - max_drop)
       을 만족하는 것만 추가
    3) 최종 페이지 오름차순
    """
    picked: List[int] = []
    seen = set()

    # 1) LLM 근거 페이지 우선
    for p in cited_pages or []:
        try:
            p = int(p)
        except Exception:
            continue
        if p in seen:
            continue
        seen.add(p)
        picked.append(p)
        if len(picked) >= max_pages:
            return sorted(picked)

    # top1이 없으면 contexts[0]에서 추정
    if top1_similarity is None:
        if contexts:
            try:
                top1_similarity = float(contexts[0].get("similarity", -1.0))
            except Exception:
                top1_similarity = -1.0
        else:
            top1_similarity = -1.0

    # 2) contexts로 보충 (관련도 컷 적용)
    #    - threshold = max(min_abs, top1 - max_drop)
    rel_threshold = max(min_abs, (top1_similarity - max_drop))

    for c in contexts or []:
        try:
            sim = float(c.get("similarity", -1.0))
            p = int(c.get("page_number"))
        except Exception:
            continue

        # ✅ 너무 낮으면 제외
        if sim < rel_threshold:
            continue

        if p in seen:
            continue

        seen.add(p)
        picked.append(p)

        if len(picked) >= max_pages:
            break

    return sorted(picked)


def is_toc_page(text: str) -> bool:
    """
    목차 페이지 휴리스틱 판정 (한국어/영문 대응)
    """
    t = (text or "").strip()
    if not t:
        return False

    low = t.lower()
    keyword = ("목차" in t) or ("table of contents" in low) or (re.search(r"\bcontents\b", low) is not None)
    if not keyword:
        return False

    dot_leader_count = len(re.findall(r"\.{3,}", t))
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    numeric_tail_lines = 0
    for ln in lines[:80]:
        if re.search(r"\d+\s*$", ln) and (len(ln) < 120):
            numeric_tail_lines += 1

    short_lines = sum(1 for ln in lines[:80] if len(ln) <= 60)

    score = 0
    if dot_leader_count >= 3:
        score += 1
    if numeric_tail_lines >= 6:
        score += 1
    if short_lines >= 25:
        score += 1

    return score >= 1


def normalize_vertical_text(text: str) -> str:
    """
    세로 OCR 결과를 가로 문장으로 정리
    예) E\\nR\\nR\\nO\\nR  -> ERROR
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if lines and all(len(l) == 1 for l in lines):
        return "".join(lines)
    return " ".join(lines)
