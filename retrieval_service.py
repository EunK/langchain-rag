from typing import List, Optional, Dict, Any, Tuple
from clients import get_openai_client, get_supabase_client
from config import Settings
from utils_text import robust_json_loads


def openai_embed(client, model: str, text: str) -> List[float]:
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def embedding_to_pgvector_str(emb: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in emb) + "]"


def retrieve_contexts(
    settings: Settings,
    question: str,
    doc_id_filter: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    oai = get_openai_client(settings.openai_api_key)
    sb = get_supabase_client(settings.supabase_url, settings.supabase_service_key)

    q_emb = openai_embed(oai, settings.embedding_model, question)
    if len(q_emb) != settings.embedding_dims:
        raise ValueError(f"Query embedding dims mismatch: got {len(q_emb)}, expected {settings.embedding_dims}")

    payload = {
        "query_embedding": embedding_to_pgvector_str(q_emb),
        "match_count": settings.top_k,
        "doc_id_filter": doc_id_filter,
    }

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
