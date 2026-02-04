import time
from typing import Tuple
import fitz  # PyMuPDF

from clients import get_openai_client, get_supabase_client
from config import Settings
from utils_text import chunk_text, is_toc_page
from retrieval_service import openai_embed, embedding_to_pgvector_str
from storage_service import supabase_upload_png


def ingest_pdf_to_supabase(settings: Settings, pdf_bytes: bytes, title: str) -> Tuple[int, int]:
    oai = get_openai_client(settings.openai_api_key)
    sb = get_supabase_client(settings.supabase_url, settings.supabase_service_key)

    doc_row = sb.table("manual_docs").insert({"title": title, "file_name": f"{title}.pdf"}).execute()
    doc_id = int(doc_row.data[0]["id"])

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_chunks = 0

    for page_index in range(doc.page_count):
        page_number = page_index + 1
        page = doc.load_page(page_index)

        text = page.get_text("text") or ""
        toc_flag = is_toc_page(text)

        pix = page.get_pixmap(dpi=160)
        png = pix.tobytes("png")

        img_path = f"{doc_id}/page_{page_number:04d}.png"
        img_url = supabase_upload_png(sb, settings.storage_bucket, img_path, png)

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

        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            continue

        rows = []
        for ci, chunk in enumerate(chunks):
            emb = openai_embed(oai, settings.embedding_model, chunk)
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
            if total_chunks % 60 == 0:
                time.sleep(0.25)

        sb.table("rag_chunks").insert(rows).execute()

    return doc_id, total_chunks
