import os
from dotenv import load_dotenv
import streamlit as st
import RAG_LLM as rag
from openai import OpenAI
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv(dotenv_path='seok.env')
api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

# Streamlit 기본 설정
st.header("compare summerized news by llm VS original news")

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if user_input := st.chat_input("질문을 입력하세요!"):
    # 사용자 메시지를 히스토리에 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG 모델 호출
    with st.chat_message("assistant"):
        try:
            response_llm = rag.rag_chain_llm.invoke(user_input)

            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    *[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    *[
                        {"role": "system", "content": response_llm.content},
                        {"role": "user", "content": response_llm.content},
                    ],
                ],
                stream=True,
            )
            response = st.write_stream(stream)

            #어시 메시지 히스토리에 추가
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"RAG 호출 중 오류 발생: {str(e)}")