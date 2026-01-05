보안 장비 AI Assistant
# 실행순서
1. image_crop.py : 이미지와 텍스트 분리하여 AI 응답으로 JOSN 파일 생성
2. build_db.py: 생성된 json 파일을 local Vector DB(chroma)에 저장
3. chat_agent.py : prompt UI의 질문 입력 UI  실행


PDF_SERVICES_CLIENT_ID=e71430df90eb4c7584dd3996b2a37e3b
PDF_SERVICES_CLIENT_SECRET=p8e--Xy8plM98QjgjCbhQf5h-J6Xsfshv0oK
OPENAI_API_KEY=
SUPABASE_URL=https://wnbfjewwrbfzkxwhowek.supabase.co/
SUPABASE_SERVICE_ROLE_KEY=sb_secret_0esilyIgHR6-D-Saym6Drg_d7Sl2xMH
