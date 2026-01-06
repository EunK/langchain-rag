import hashlib
import streamlit as st

from config import load_settings
from clients import get_openai_client
from ocr_service import extract_text_from_image_gpt41mini
from ingest_service import ingest_pdf_to_supabase
from retrieval_service import retrieve_contexts, list_docs, get_page_image_url
from answer_service import openai_answer_with_rag
from storage_service import delete_doc_and_assets
from utils_text import is_refusal_answer, merge_pages_cited_then_search


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

    # 채팅 히스토리
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ✅ OCR / draft 상태
    if "draft_question" not in st.session_state:
        st.session_state.draft_question = ""
    if "ocr_image_signature" not in st.session_state:
        st.session_state.ocr_image_signature = None  # 마지막 OCR 수행한 이미지 식별자
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""

    # -------------------------
    # 이미지 업로드 → 자동 OCR (새 이미지일 때만 1회)
    # -------------------------
    st.markdown("### 📷 이미지 업로드 (업로드 시 자동 OCR, 질문 전송과 무관)")
    img_file = st.file_uploader(
        "장비 화면 이미지를 업로드하세요 (png/jpg/jpeg)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    if img_file:
        img_bytes = img_file.getvalue()
        mime = img_file.type or "image/png"

        # 미리보기
        st.image(img_bytes, caption="업로드한 이미지", width=350)

        # ✅ 내용 기반 시그니처(해시): rerun(질문 전송)에도 동일 이미지면 OCR 재실행 안 함
        digest = hashlib.sha256(img_bytes).hexdigest()
        image_signature = f"{digest}"

        # ✅ 새 이미지일 때만 OCR 실행
        if st.session_state.ocr_image_signature != image_signature:
            with st.spinner("이미지에서 문자 추출 중 (gpt-4.1-mini)..."):
                oai = get_openai_client(settings.openai_api_key)
                ocr_text = extract_text_from_image_gpt41mini(oai, img_bytes, mime)

            st.session_state.ocr_image_signature = image_signature
            st.session_state.ocr_text = (ocr_text or "").strip()

            if st.session_state.ocr_text:
                # ✅ OCR 결과를 질문창으로 보내기(자동 질문 전송 X)
                st.session_state.draft_question = st.session_state.ocr_text
                st.success("OCR 완료: 질문 입력창에 반영되었습니다. 필요하면 수정 후 전송하세요.")
            else:
                st.warning("OCR 결과가 비어있습니다. 이미지 해상도/선명도를 확인해 주세요.")

    # -------------------------
    # 질문 입력 / 전송 (OCR과 무관: 질문창 내용만 전송)
    # -------------------------
    question = st.text_area(
        "질문 입력 (OCR 결과가 있으면 자동으로 표시됩니다. 수정 후 전송하세요.)",
        value=st.session_state.draft_question,
        height=120,
    )

    col1, col2, space = st.columns([2, 0.8, 5])
    with col1:
        send = st.button(
            "질문 전송", 
            type="primary", 
            disabled=not question.strip() if question else True,
            use_container_width=True
        )

    with col2:
        voice_btn = st.button("🎙️", use_container_width=True, key="voice_btn_btn")

    with space:
        # 아무것도 작성하지 않으면 빈 공간으로 남습니다.
        pass

    if send:
        # 전송은 질문창 내용만 사용 (OCR 재실행과 무관)
        st.session_state.draft_question = ""

        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("검색 및 답변 생성 중..."):
                contexts, top1_similarity = retrieve_contexts(settings, question, doc_id_filter=doc_id_filter)
                st.caption(f"top1 similarity = {top1_similarity:.3f} (threshold={settings.similarity_threshold:.2f})")

                out_of_scope = (not contexts) or (top1_similarity < settings.similarity_threshold)
                cited_pages = []

                if out_of_scope:
                    answer = "문서에 존재하지 않습니다."
                else:
                    oai = get_openai_client(settings.openai_api_key)
                    out = openai_answer_with_rag(oai, settings.chat_model, question, contexts)
                    answer = out["answer"]
                    cited_pages = out.get("cited_pages", [])

                    # 보수적으로 한 번 더 차단 (애매한 경우)
                    if ("문서에 존재하지 않습니다" not in answer) and (top1_similarity < (settings.similarity_threshold + 0.02)):
                        answer = "문서에 존재하지 않습니다."
                        cited_pages = []

                st.markdown(answer)

                # 관련 페이지: 거절답변이면 절대 표시하지 않음
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

                # 관련 페이지 3+3 (최대 6)
                if resolved_doc_id and related_pages:
                    st.caption("관련 페이지 (최대 6페이지, 페이지 순)")

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

                    row2 = related_pages[3:6]
                    if row2:
                        cols2 = st.columns(3)
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
