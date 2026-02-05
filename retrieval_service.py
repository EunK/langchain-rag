import google.generativeai as genai
from typing import List, Optional, Dict, Any, Tuple
from clients import init_gemini, get_supabase_client  # 이전 단계에서 만든 init_gemini 사용
from config import Settings
from utils_text import robust_json_loads

def gemini_embed(model: str, text: str) -> List[float]:
    """
    Gemini 기반 임베딩 생성
    model 예시: 'models/text-embedding-004'
    """
    # Gemini는 task_type을 지정하여 검색 품질을 높일 수 있습니다.
    result = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

def embedding_to_pgvector_str(emb: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in emb) + "]"

def retrieve_contexts(
    settings: Settings,
    question: str,
    doc_id_filter: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    # 1. Gemini 초기화 (환경변수 키 사용)
    init_gemini(settings.gemini_api_key)
    sb = get_supabase_client(settings.supabase_url, settings.supabase_service_key)

    # 2. Gemini 임베딩 생성
    # settings.embedding_model은 이제 'models/text-embedding-004'여야 합니다.
    q_emb = gemini_embed(settings.embedding_model, question)
    
    # 벡터 차원 검증 (Gemini 004 모델 기준 보통 768)
    if len(q_emb) != settings.embedding_dims:
        raise ValueError(
            f"Query embedding dims mismatch: got {len(q_emb)}, expected {settings.embedding_dims}. "
            f"임베딩 모델이 바뀌었다면 DB 내의 벡터 데이터도 재생성해야 합니다."
        )

    payload = {
        "query_embedding": embedding_to_pgvector_str(q_emb),
        "match_count": settings.top_k,
        "doc_id_filter": doc_id_filter,
    }

    # Supabase RPC 호출
    res = sb.rpc("match_rag_chunks_v3", payload).execute()
    rows = res.data or []

    contexts = []
    top1_similarity = -1.0

    for i, r in enumerate(rows):
        sim = float(r.get("similarity", -1.0))
        if i == 0:
            top1_similarity = sim

        contexts.append(
            {
                "id": r["id"],
                "doc_id": r["doc_id"],
                "page_number": r["page_number"],
                "chunk_index": r["chunk_index"],
                "content": r["content"],
                "similarity": sim,
            }
        )

    return contexts, top1_similarity

# --- 이미지 URL 가져오기 및 문서 목록 함수는 모델과 무관하므로 그대로 유지 ---
def get_page_image_url(settings: Settings, doc_id: int, page_number: int) -> Optional[str]:
    sb = get_supabase_client(settings.supabase_url, settings.supabase_service_key)
    res = (
        sb.table("manual_pages")
        .select("image_url,is_toc")
        .eq("doc_id", doc_id)
        .eq("page_number", page_number)
        .limit(1)
        .execute()
    )
    if res.data:
        if bool(res.data[0].get("is_toc")) is True:
            return None
        return res.data[0].get("image_url")
    return None

def list_docs(settings: Settings) -> List[Dict[str, Any]]:
    sb = get_supabase_client(settings.supabase_url, settings.supabase_service_key)
    res = sb.table("manual_docs").select("id,title,created_at").order("created_at", desc=True).execute()
    return res.data or []