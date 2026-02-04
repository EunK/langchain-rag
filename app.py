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
from PIL import Image
from io import BytesIO  # ✅ 추가

from audio_recorder_streamlit import audio_recorder
import os
import tempfile
from process_rag_query import process_rag_query
from render import render_related_pages, get_related_pages

st.set_page_config(page_title="NexOps-가장 명확한 근거, 가장 빠른 현장 조치", layout="wide")
settings = load_settings()

st.title("🛡️ NexOps for Security")

if not settings.openai_api_key or not settings.supabase_url or not settings.supabase_service_key:
    st.warning(
        "환경변수가 필요합니다!\n\n"
        "- OPENAI_API_KEY\n"
        "- SUPABASE_URL\n"
        "- SUPABASE_SERVICE_ROLE_KEY\n"
    )
    st.stop()

mode = st.sidebar.radio("메뉴", ["AI 현장 가이드", "지식 자산 관리"])

st.sidebar.markdown("---")
settings.similarity_threshold = st.sidebar.slider(
    "Out-of-scope 유사도 임계치",
    min_value=0.00,
    max_value=1.00,
    value=float(settings.similarity_threshold),
    step=0.01,
    help="top1 similarity가 이 값보다 작으면 '문서에 존재하지 않습니다.'",
)

# [추가] 이미지 축소 최대 px 슬라이더 (OCR/해시/미리보기 공통 적용)
resize_max_px = st.sidebar.slider(
    "이미지 축소 최대 px",
    min_value=512,
    max_value=2048,
    value=1024,
    step=64,
    help="업로드 이미지의 긴 변을 이 값 이하로 축소합니다. (OCR 비용/속도 최적화)",
)


# -------------------------
# Admin
# -------------------------
if mode == "지식 자산 관리":
    st.subheader("매뉴얼 업로드 및 AI 지식 엔진 구축")

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
    st.subheader("현장 질문톡")

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

   

    # ✅ OCR / draft 상태
    if "draft_question" not in st.session_state:
        st.session_state.draft_question = "질문 입력 (OCR 결과가 있으면 자동으로 표시됩니다. 수정 후 전송하세요.)"
    if "ocr_image_signature" not in st.session_state:
        st.session_state.ocr_image_signature = None  # 마지막 OCR 수행한 이미지 식별자
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "last_audio_bytes" not in st.session_state:
        st.session_state.last_audio_bytes = None
    if "transcription_result" not in st.session_state:
        st.session_state.transcription_result = None
    if "finish_voice" not in st.session_state:
        st.session_state.finish_voice = False

     # 채팅 히스토리
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("pages"):
                render_related_pages(msg.get("pages"))



    # -------------------------
    # 질문 입력 / 전송 (OCR과 무관: 질문창 내용만 전송)
    # -------------------------
    print(f"st.session_state.draft_question:{st.session_state.draft_question}")
    prompt = st.chat_input(
       st.session_state.draft_question
    )

    col1, col2, space = st.columns([0.07, 0.33, 0.6], vertical_alignment ="bottom")
    with col1:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=2.0,
            key="audio_recorder"
        )

    with col2:
        img_file = st.file_uploader(
                "📷 장비 화면 이미지를 업로드 (업로드 시 자동 OCR, 질문 전송과 무관)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
            )
        # col3, col4, col5 = st.columns([6, 3, 2])
        # # -------------------------
        # # 이미지 업로드 → 자동 OCR (새 이미지일 때만 1회)
        # # -------------------------
        # with col3:
        #     st.markdown("### 📷 이미지 업로드 (업로드 시 자동 OCR, 질문 전송과 무관)")
        # with col4:
        #     img_file = st.file_uploader(
        #         "📷 장비 화면 이미지를 업로드 (업로드 시 자동 OCR, 질문 전송과 무관)",
        #         type=["png", "jpg", "jpeg"],
        #         accept_multiple_files=False,
        #     )
        # with col5:
        #     pass
    with space:
            # 아무것도 작성하지 않으면 빈 공간으로 남습니다.
        pass
    # print("0 start")
    # if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
    #      print("1 녹음 완료")
    # elif st.session_state.last_audio_bytes:
    #     print("1 녹음 완료")

    # 변환 처리
    if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
        st.session_state.last_audio_bytes = audio_bytes
        st.session_state.transcription_result = None  # 이전 결과 초기화
        print("1 녹음 저장")
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(st.session_state.last_audio_bytes)
            wav_path = tmp_file.name
        try:
            # Whisper API 호출
            with st.spinner("🤖 음성을 텍스트로 변환 중..."):
                with open(wav_path, "rb") as audio_file:
                    client = get_openai_client(settings.openai_api_key)
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="ko",
                        response_format="text"
                    )
        
            # 결과 저장
            st.session_state.transcription_result = transcript
            print(f"음성변환: {transcript}")
        
        except Exception as e:
            st.error(f"❌ 변환 실패: {str(e)}")
            st.exception(e)
        
        finally:
            # 임시 파일 삭제
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass
    
    # 5. 실제 질문 결정
    final_prompt = prompt or st.session_state.transcription_result 
    if final_prompt:
        print("질문")   
        st.session_state.draft_question = ""

        st.session_state.chat.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        with st.chat_message("assistant"):
            with st.spinner("검색 및 답변 생성 중..."):
                result = process_rag_query(settings, final_prompt, doc_id_filter)
                answer = result["answer"]
                related_pages = result["related_pages"]
                resolved_doc_id = result["resolved_doc_id"]
                top1_similarity = result["top1_similarity"]
                # # 유사도 정보 표시
                # st.caption(f"top1 similarity = {top1_similarity:.3f} (threshold={settings.similarity_threshold:.2f})")
                
                # 답변 출력
                # st.markdown(answer)
            pages = get_related_pages(
                settings=settings,
                resolved_doc_id=resolved_doc_id,
                related_pages=related_pages
            )
            # render_related_pages(pages)
        st.session_state.chat.append({
            "role": "assistant",
            "content": answer,
            "pages": pages
        })
        st.session_state.draft_question = final_prompt
        st.session_state.transcription_result = None
        st.rerun()
    
    if img_file: 
        print("이미지")   
        img_bytes = img_file.getvalue()
        mime = img_file.type or "image/png"
        if len(st.session_state.ocr_text) > 0:
            question = st.text_input(
                "OCR 질문",
                value=st.session_state.ocr_text
            )

        # 미리보기
        # col1, col2 = st.columns([1, 7], gap="medium")
        # with col1:
        #     st.image(img_bytes, caption="업로드한 이미지")
        # with col2:
        #     if len(st.session_state.ocr_text) > 0:
        #         question = st.text_input(
        #             "OCR 질문",
        #             value=st.session_state.ocr_text
        #         )
        #         # if st.button("이 질문으로 전송"):
        #         #     print("전송")
        #     else:
        #         pass
                

        # ✅ 최대 1024px로 자동 축소 (비율 유지)
        try:
            pil_img = Image.open(BytesIO(img_bytes))
            pil_img.thumbnail((resize_max_px, resize_max_px), Image.LANCZOS)

            buf = BytesIO()
            # 원본 포맷을 최대한 유지 (없으면 PNG)
            save_format = (pil_img.format or "PNG").upper()
            if save_format not in ("PNG", "JPEG", "JPG", "WEBP"):
                save_format = "PNG"

            # JPEG로 저장할 때는 RGB 필요할 수 있음
            if save_format in ("JPEG", "JPG") and pil_img.mode in ("RGBA", "P"):
                pil_img = pil_img.convert("RGB")

            pil_img.save(buf, format=save_format)
            img_bytes = buf.getvalue()  # ✅ 이후 로직(OCR/미리보기/해시)에 축소본을 사용
        except Exception:
            # 축소 실패 시 원본 유지
            pass

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
                st.rerun()
            else:
                st.warning("OCR 결과가 비어있습니다. 이미지 해상도/선명도를 확인해 주세요.")

