보안 장비 AI Assistant
# 실행순서
1. image_crop.py : 이미지와 텍스트 분리하여 AI 응답으로 JOSN 파일 생성
2. build_db.py: 생성된 json 파일을 local Vector DB(chroma)에 저장
3. chat_agent.py : prompt UI의 질문 입력 UI  실행
