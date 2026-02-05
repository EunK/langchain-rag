import time
import google.generativeai as genai
from typing import Tuple
import fitz  # PyMuPDF

from clients import get_gemini_client, get_supabase_client
from config import Settings
from utils_text import chunk_text, is_toc_page
# retrieval_service에서 gemini_embed를 가져오거나 아래에 직접 정의한 것을 사용합니다.
from retrieval_service import gemini_embed, embedding_to_pgvector_str
from storage_service import supabase_upload_png


def ingest_pdf_to_supabase(settings: Settings, pdf_bytes: bytes, title: str) -> Tuple[int, int]:
    # 1. Gemini 초기화 및 Supabase 클라이언트 준비
    get_gemini_client(settings.google_api_key) # 오타 수정: googl_api_key -> google_api_key
    sb = get_supabase_client(settings.supabase_url, settings.supabase_service_key)

    # 2. 문서 레코드 생성
    doc_row = sb.table("manual_docs").insert({"title": title, "file_name": f"{title}.pdf"}).execute()
    doc_id = int(doc_row.data[0]["id"])

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_chunks = 0

    for page_index in range(doc.page_count):
        page_number = page_index + 1
        page = doc.load_page(page_index)

        # 텍스트 추출 및 목차 여부 확인
        text = page.get_text("text") or ""
        toc_flag = is_toc_page(text)

        # 페이지 이미지화 및 업로드 (이 로직은 동일)
        pix = page.get_pixmap(dpi=160)
        png = pix.tobytes("png")
        img_path = f"{doc_id}/page_{page_number:04d}.png"
        img_url = supabase_upload_png(sb, settings.storage_bucket, img_path, png)

        # 페이지 정보 저장
        sb.table("manual_pages").upsert(
            {
                "doc_id": doc_id,
                "page_number": page_number,
                "image_path": img_path,
                "image_url": img_url,
                "is_toc": toc_flag,
            },
            on_conflict="doc_id,page_number",
        ).execute()

        # 청크 분할
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            continue

        rows = []
        for ci, chunk in enumerate(chunks):
            # 3. Gemini 임베딩 생성 (task_type="retrieval_document" 지정)
            # retrieval_service.py에 정의된 gemini_embed를 활용하거나 직접 호출
            result = genai.embed_content(
                model=settings.embedding_model,
                content=chunk,
                task_type="retrieval_document", # 문서 저장용 최적화
                title=title # (선택사항) 문서 제목을 넣으면 검색 품질이 더 좋아집니다.
            )
            emb = result['embedding']

            if len(emb) != settings.embedding_dims:
                raise ValueError(f"Embedding dims mismatch: got {len(emb)}, expected {settings.embedding_dims}")

            rows.append(
                {
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "chunk_index": ci,
                    "content": chunk,
                    "embedding": embedding_to_pgvector_str(emb),
                    "is_toc": toc_flag,
                }
            )
            total_chunks += 1
            
            # Gemini Free Tier(무료 버전) 사용 시 Rate Limit을 고려한 지연
            if total_chunks % 15 == 0: # 무료 버전은 분당 요청 제한이 엄격하므로 좀 더 넉넉히 쉽니다.
                time.sleep(1.0)

        # 해당 페이지의 모든 청크를 한 번에 인서트
        if rows:
            sb.table("rag_chunks").insert(rows).execute()

    return doc_id, total_chunks