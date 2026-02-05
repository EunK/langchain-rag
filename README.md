# NexOps 보안장비 메뉴얼 AI Assistant  
## 실행순서  
> ### 1. image_crop.py : 이미지와 텍스트 분리하여 AI 응답으로 JOSN 파일 생성  
> ### 2. build_db.py: 생성된 json 파일을 local Vector DB(chroma)에 저장  
> ### 3. chat_agent.py : prompt UI의 질문 입력 UI  실행  
---
##  환경설정
### .evv 에 아래 값 필요합니다.  
### PDF_SERVICES_CLIENT_ID=  
### PDF_SERVICES_CLIENT_SECRET=  
### GOOGL_API_KEY=  
### SUPABASE_URL=https:/...  
### SUPABASE_SERVICE_ROLE_KEY=...  
--- Vector DB SQL
```
Supabase
-- 1. 벡터 검색을 위한 pgvector 확장 기능 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 문서 마스터 테이블 (manual_docs)
-- PDF 문서의 기본 정보를 저장합니다.
CREATE TABLE IF NOT EXISTS public.manual_docs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title TEXT NOT NULL,
    file_name TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- 3. 페이지 정보 테이블 (manual_pages)
-- 각 페이지의 이미지 URL과 목차(TOC) 여부를 관리합니다.
CREATE TABLE IF NOT EXISTS public.manual_pages (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_id BIGINT REFERENCES public.manual_docs(id) ON DELETE CASCADE,
    page_number INT NOT NULL,
    image_path TEXT,
    image_url TEXT,
    is_toc BOOLEAN DEFAULT FALSE,
    UNIQUE(doc_id, page_number) -- 동일 문서 내 페이지 번호 중복 방지
);

-- 4. RAG 청크 및 벡터 데이터 테이블 (rag_chunks)
-- Gemini (text-embedding-004)의 768차원 벡터를 저장합니다.
CREATE TABLE IF NOT EXISTS public.rag_chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_id BIGINT REFERENCES public.manual_docs(id) ON DELETE CASCADE,
    page_number INT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768), -- Gemini 모델에 맞춘 768차원 설정
    is_toc BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- 5. 벡터 유사도 검색을 위한 RPC 함수 (match_rag_chunks_v3)
-- Python 코드의 sb.rpc("match_rag_chunks_v3", ...) 와 매칭됩니다.
CREATE OR REPLACE FUNCTION public.match_rag_chunks_v3 (
  query_embedding VECTOR(768),
  match_count INT DEFAULT 5,
  doc_id_filter BIGINT DEFAULT NULL
)
RETURNS TABLE (
  id BIGINT,
  doc_id BIGINT,
  page_number INT,
  chunk_index INT,
  content TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    rc.id,
    rc.doc_id,
    rc.page_number,
    rc.chunk_index,
    rc.content,
    1 - (rc.embedding <=> query_embedding) AS similarity -- 코사인 유사도 계산
  FROM public.rag_chunks rc
  WHERE (doc_id_filter IS NULL OR rc.doc_id = doc_id_filter)
  ORDER BY rc.embedding <=> query_embedding -- 가장 가까운 벡터 순으로 정렬
  LIMIT match_count;
END;
$$;

-- 6. 검색 성능 향상을 위한 인덱스 (선택 사항)
-- 데이터가 많아질 경우 아래 인덱스를 활성화하여 검색 속도를 높일 수 있습니다.
-- CREATE INDEX ON public.rag_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```