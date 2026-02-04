import streamlit as st
from retrieval_service import  get_page_image_url

def render_related_pages(pages):
    if len(pages) < 1:
        return
    
    # st.caption("관련 페이지 (최대 6페이지, 페이지 순)")
    with st.expander("관련 페이지가 있습니다.", icon="➕"):
        # 3개씩 끊어서 행 구성
        for row_start in range(0, len(pages), 3):
            row_pages = pages[row_start:row_start + 3]
            cols = st.columns(3)

            for idx, item in enumerate(row_pages):
                with cols[idx]:
                    p = item["page"]
                    url = item["url"]

                    if url == "":
                        st.write(f"p.{p} 이미지 없음")
                    else:
                        st.image(url, caption=f"p.{p}", width="stretch")


def get_related_pages(settings, resolved_doc_id, related_pages, max_pages=6):
    """
    관련 페이지 이미지를 최대 max_pages까지 3열 그리드로 표시하고,
    [{"page": p, "url": url}, ...] 형태로 반환한다.
    url 이 없는 경우 "" 로 대체한다.
    """
    results = []

    if not resolved_doc_id or not related_pages:
        return results

    # st.caption(f"관련 페이지 (최대 {max_pages}페이지, 페이지 순)")

    pages = related_pages[:max_pages]

    for row_start in range(0, len(pages), 3):
        row_pages = pages[row_start:row_start + 3]
        # cols = st.columns(3)

        for idx, p in enumerate(row_pages):
            url = get_page_image_url(settings, resolved_doc_id, int(p))
            if url:
                # st.image(url, caption=f"p.{p}", width="stretch")
                results.append({"page": p, "url": url})
            else:
                # st.write(f"p.{p} 이미지 없음")
                results.append({"page": p, "url": ""})
                

    return results
