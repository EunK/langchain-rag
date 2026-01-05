import os
import json
import time
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import streamlit as st
import fitz  # PyMuPDF

from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv


# =========================
# Config
# =========================

@dataclass
class Settings:
    openai_api_key: str
    supabase_url: str
    supabase_service_key: str
    storage_bucket: str = "manual-pages"

    # Retrieval
    top_k: int = 10

    # UI slider default = 0.00 (keep previous behavior)
    similarity_threshold: float = 0.00

    # Related pages
    max_related_pages: int = 5

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Models
    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"

    # text-embedding-3-small default dims
    embedding_dims: int = 1536


def _ensure_trailing_slash(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    return url if url.endswith("/") else (url + "/")

load_dotenv()
def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_service_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    )

# =========================
# Clients
# =========================

@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


@st.cache_resource
def get_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)


# =========================
# Utilities
# =========================

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


def openai_embed(client: OpenAI, model: str, text: str) -> List[float]:
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def embedding_to_pgvector_str(emb: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in emb) + "]"


def is_refusal_answer(answer: str) -> bool:
    if not answer:
        return True
    return "문서에 존재하지 않습니다" in answer.strip()


def merge_pages_cited_then_search(
    cited_pages: List[int],
    contexts: List[Dict[str, Any]],
    max_pages: int
) -> List[int]:
    """
    1) cited_pages 우선
    2) 부족하면 검색 결과 contexts에서 유니크 페이지로 보충
    3) 최종 페이지 오름차순
    """
    picked: List[int] = []
    seen = set()

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

    for c in contexts or []:
        p = int(c["page_number"])
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
    - '목차' 또는 'contents/table of contents' 포함 + 목차 특유 패턴(도트 리더/짧은 라인 반복/페이지번호 나열) 중 일부
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
        "너는 장비 매뉴얼 PDF를 기반으로만 답하는 고객지원 챗봇이다.\n"
        "규칙:\n"
        "1) 아래 제공된 '매뉴얼 발췌'에 없는 정보는 절대 추측하지 말고, 반드시 '문서에 존재하지 않습니다.' 라고 답하라.\n"
        "2) 답변은 한국어로, 간결하되 사용자가 바로 실행할 수 있게 단계형으로 작성하라.\n"
        "3) 답변에 근거가 된 페이지 번호를 cited_pages 배열로 반드시 포함하라.\n"
        "4) 출력은 JSON 하나로만: {\"answer\": string, \"cited_pages\": number[]}\n"
    )

    user = f"질문:\n{question}\n\n매뉴얼 발췌:\n{ctx_text}\n"

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text_out = resp.output_text
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        text_out = resp.choices[0].message.content or ""

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


# =========================
# Storage helpers
# =========================

def ensure_bucket_exists(sb: Client, bucket: str, public: bool = True) -> None:
    try:
        buckets = sb.storage.list_buckets()
        exists = any(b.get("name") == bucket for b in buckets)
        if not exists:
            sb.storage.create_bucket(bucket, public=public)
        return
    except Exception:
        pass

    try:
        sb.storage.create_bucket(bucket, public=public)
    except Exception:
        return


def supabase_upload_png(sb: Client, bucket: str, path: str, png_bytes: bytes) -> str:
    ensure_bucket_exists(sb, bucket, public=True)
    try:
        sb.storage.from_(bucket).upload(
            path=path,
            file=png_bytes,
            file_options={"content-type": "image/png", "upsert": "true"},
        )
    except Exception:
        ensure_bucket_exists(sb, bucket, public=True)
        sb.storage.from_(bucket).upload(
            path=path,
            file=png_bytes,
            file_options={"content-type": "image/png", "upsert": "true"},
        )
    return sb.storage.from_(bucket).get_public_url(path)


def _chunks(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def delete_doc_and_assets(settings: Settings, doc_id: int) -> Dict[str, Any]:
    """
    doc_id의:
    - Storage(이미지) 삭제
    - DB(rag_chunks, manual_pages, manual_docs) 삭제
    """
    sb = get_supabase_client(settings.supabase_url, settings.supabase_service_key)

    # 1) 이미지 path 수집
    pages_res = (
        sb.table("manual_pages")
        .select("image_path")
        .eq("doc_id", doc_id)
        .execute()
    )
    image_paths = [r["image_path"] for r in (pages_res.data or []) if r.get("image_path")]

    # 2) Storage 삭제 (배치)
    storage_deleted = 0
    storage_failed: List[str] = []
    if image_paths:
        for batch in _chunks(image_paths, 100):
            try:
                sb.storage.from_(settings.storage_bucket).remove(batch)
                storage_deleted += len(batch)
            except Exception:
                storage_failed.extend(batch)

    # 3) DB 삭제 (순서 중요: child -> parent)
    # rag_chunks
    try:
        sb.table("rag_chunks").delete().eq("doc_id", doc_id).execute()
    except Exception as e:
        return {"ok": False, "error": f"rag_chunks delete failed: {e}", "storage_deleted": storage_deleted, "storage_failed": storage_failed}

    # manual_pages
    try:
        sb.table("manual_pages").delete().eq("doc_id", doc_id).execute()
    except Exception as e:
        return {"ok": False, "error": f"manual_pages delete failed: {e}", "storage_deleted": storage_deleted, "storage_failed": storage_failed}

    # manual_docs
    try:
        sb.table("manual_docs").delete().eq("id", doc_id).execute()
    except Exception as e:
        return {"ok": False, "error": f"manual_docs delete failed: {e}", "storage_deleted": storage_deleted, "storage_failed": storage_failed}

    return {"ok": True, "storage_deleted": storage_deleted, "storage_failed": storage_failed}


# =========================
# Ingest
# =========================

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


# =========================
# Retrieval
# =========================

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

    # ✅ DB 레벨에서 is_toc=false만 반환되도록 RPC가 필터링해야 합니다.
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


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="PDF 매뉴얼 RAG 챗봇", layout="wide")
settings = load_settings()

st.title("📘 PDF 매뉴얼 RAG 챗봇 (Supabase + OpenAI)")

if not settings.openai_api_key or not settings.supabase_url or not settings.supabase_service_key:
    st.warning(
        "환경변수가 필요합니다.\n\n"
        "- OPENAI_API_KEY\n"
        "- SUPABASE_URL\n"
        "- SUPABASE_SERVICE_ROLE_KEY\n"
    )
    st.stop()

mode = st.sidebar.radio("메뉴", ["관리자: PDF 업로드/적재", "사용자: 챗봇"])

st.sidebar.markdown("---")
settings.similarity_threshold = st.sidebar.slider(
    "Out-of-scope 유사도 임계치(높을수록 엄격)",
    min_value=0.00,
    max_value=1.00,
    value=float(settings.similarity_threshold),
    step=0.01,
    help="top1 similarity가 이 값보다 작으면 '문서에 존재하지 않습니다.'",
)


# -------------------------
# Admin
# -------------------------
if mode == "관리자: PDF 업로드/적재":
    st.subheader("관리자: PDF 업로드 및 RAG 적재")

    title = st.text_input("문서 제목(예: 장비A_매뉴얼)", value="")
    pdf = st.file_uploader("PDF 업로드", type=["pdf"])

    if st.button("적재 실행", type="primary", disabled=not (title and pdf)):
        with st.spinner("PDF를 페이지별로 처리하고, 임베딩을 생성하여 Supabase에 저장 중..."):
            pdf_bytes = pdf.read()
            doc_id, total_chunks = ingest_pdf_to_supabase(settings, pdf_bytes, title)
        st.success(f"완료! doc_id={doc_id}, total_chunks={total_chunks}")
        st.info("※ 목차 제외(DB레벨)는 is_toc 태깅이 필요하므로, 이 방식 적용 후에는 재적재가 반영됩니다.")

    st.divider()
    st.subheader("적재된 문서 목록")
    docs = list_docs(settings)
    if not docs:
        st.info("아직 적재된 문서가 없습니다.")
    else:
        for d in docs:
            st.write(f"- #{d['id']} | {d['title']} | {d['created_at']}")

    # ✅ 문서 삭제 UI
    st.divider()
    st.subheader("문서 삭제 (DB + Storage 이미지)")

    docs = list_docs(settings)
    if not docs:
        st.info("삭제할 문서가 없습니다.")
    else:
        doc_map = {f"#{d['id']} - {d['title']}": int(d["id"]) for d in docs}
        sel_label = st.selectbox("삭제할 문서 선택", options=list(doc_map.keys()))
        del_doc_id = doc_map[sel_label]

        confirm = st.checkbox("정말 삭제합니다. (DB + Storage 이미지까지 삭제됨)", value=False)
        if st.button("선택 문서 삭제", type="secondary", disabled=not confirm):
            with st.spinner(f"doc_id={del_doc_id} 삭제 중..."):
                result = delete_doc_and_assets(settings, del_doc_id)

            if result.get("ok"):
                st.success(f"삭제 완료: doc_id={del_doc_id}")
                st.write(f"- Storage 삭제: {result.get('storage_deleted', 0)}개")
                failed = result.get("storage_failed", [])
                if failed:
                    st.warning(f"Storage 삭제 실패 {len(failed)}개 (권한/경로 확인 필요)")
                    st.text("\n".join(failed[:50]))
            else:
                st.error(f"삭제 실패: {result.get('error')}")

# -------------------------
# Chatbot
# -------------------------
else:
    st.subheader("사용자: 매뉴얼 Q&A")

    docs = list_docs(settings)
    doc_options = [{"id": None, "title": "전체 문서(모든 매뉴얼)"}] + [
        {"id": int(d["id"]), "title": f"#{d['id']} - {d['title']}"}
        for d in docs
    ]
    selected = st.selectbox(
        "검색 범위(문서 선택)",
        options=doc_options,
        format_func=lambda x: x["title"],
        index=0,
    )
    doc_id_filter = selected["id"]

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("장비 사용/에러/설치 방법을 질문하세요...")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("검색 및 답변 생성 중..."):
                contexts, top1_similarity = retrieve_contexts(settings, question, doc_id_filter=doc_id_filter)
                st.caption(f"top1 similarity = {top1_similarity:.3f} (threshold={settings.similarity_threshold:.2f})")

                out_of_scope = (not contexts) or (top1_similarity < settings.similarity_threshold)

                cited_pages: List[int] = []

                if out_of_scope:
                    answer = "문서에 존재하지 않습니다."
                else:
                    oai = get_openai_client(settings.openai_api_key)
                    out = openai_answer_with_rag(oai, settings.chat_model, question, contexts)
                    answer = out["answer"]
                    cited_pages = out.get("cited_pages", [])

                    if ("문서에 존재하지 않습니다" not in answer) and (top1_similarity < (settings.similarity_threshold + 0.02)):
                        answer = "문서에 존재하지 않습니다."
                        cited_pages = []

                st.markdown(answer)

                if is_refusal_answer(answer):
                    related_pages = []
                    resolved_doc_id = None
                else:
                    related_pages = merge_pages_cited_then_search(
                        cited_pages=cited_pages,
                        contexts=contexts,
                        max_pages=settings.max_related_pages,
                    )

                    if doc_id_filter is not None:
                        resolved_doc_id = doc_id_filter
                    else:
                        resolved_doc_id = int(contexts[0]["doc_id"]) if contexts else None

                # ✅ 관련 페이지 최대 5장 (3 + 2 레이아웃)
                if resolved_doc_id and related_pages:
                    st.caption("관련 페이지 (최대 5페이지, 페이지 순)")

                    # 1줄: 최대 3개
                    row1 = related_pages[:3]
                    cols1 = st.columns(3)
                    for idx in range(3):
                        with cols1[idx]:
                            if idx < len(row1):
                                p = row1[idx]
                                url = get_page_image_url(settings, resolved_doc_id, int(p))
                                if url:
                                    st.image(url, caption=f"p.{p}", width="stretch")
                                else:
                                    st.write(f"p.{p} 이미지 없음")

                    # 2줄: 나머지(최대 2개)
                    row2 = related_pages[3:5]
                    if row2:
                        cols2 = st.columns(3)  # 가운데 정렬 느낌(2개만 쓰고 1개는 비움)
                        for idx in range(3):
                            with cols2[idx]:
                                if idx < len(row2):
                                    p = row2[idx]
                                    url = get_page_image_url(settings, resolved_doc_id, int(p))
                                    if url:
                                        st.image(url, caption=f"p.{p}", width="stretch")
                                    else:
                                        st.write(f"p.{p} 이미지 없음")

        st.session_state.chat.append({"role": "assistant", "content": answer})
