from retrieval_service import retrieve_contexts
from clients import get_openai_client
from answer_service import openai_answer_with_rag
from utils_text import is_refusal_answer, merge_pages_cited_then_search

def process_rag_query(settings, question, doc_id_filter=None):
    """
    RAG 검색, 답변 생성, 임계값 검증 및 관련 페이지 추출을 처리하는 핵심 로직
    """
    # 1. 검색 (Retrieve)
    contexts, top1_similarity = retrieve_contexts(settings, question, doc_id_filter=doc_id_filter)
    
    # 2. 범위 외 확인 (Out of Scope)
    out_of_scope = (not contexts) or (top1_similarity < settings.similarity_threshold)
    
    cited_pages = []
    answer = ""
    
    if out_of_scope:
        answer = "문서에 존재하지 않습니다."
    else:
        # 3. 답변 생성 (Generate)
        oai = get_openai_client(settings.openai_api_key)
        out = openai_answer_with_rag(oai, settings.chat_model, question, contexts)
        answer = out["answer"]
        cited_pages = out.get("cited_pages", [])

        # 4. 보수적 검증 (Threshold + 0.02)
        if ("문서에 존재하지 않습니다" not in answer) and (top1_similarity < (settings.similarity_threshold + 0.02)):
            answer = "문서에 존재하지 않습니다."
            cited_pages = []

    # 5. 관련 페이지 및 문서 ID 정리
    if is_refusal_answer(answer):
        related_pages = []
        resolved_doc_id = None
    else:
        related_pages = merge_pages_cited_then_search(
            cited_pages=cited_pages,
            contexts=contexts,
            max_pages=settings.max_related_pages,
            top1_similarity=top1_similarity,
            min_abs=0.35,
            max_drop=0.08,
        )
        resolved_doc_id = (
            doc_id_filter if doc_id_filter is not None 
            else (int(contexts[0]["doc_id"]) if contexts else None)
        )

    return {
        "answer": answer,
        "related_pages": related_pages,
        "resolved_doc_id": resolved_doc_id,
        "top1_similarity": top1_similarity
    }